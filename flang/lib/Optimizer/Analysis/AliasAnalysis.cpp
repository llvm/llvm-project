//===- AliasAnalysis.cpp - Alias Analysis for FIR  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace mlir;

#define DEBUG_TYPE "fir-alias-analysis"

llvm::cl::opt<bool> supportCrayPointers(
    "unsafe-cray-pointers",
    llvm::cl::desc("Support Cray POINTERs that ALIAS with non-TARGET data"),
    llvm::cl::init(false));

// Inspect for value-scoped Allocate effects and determine whether
// 'result' is a new allocation. Returns SourceKind::Allocate if a
// MemAlloc effect is attached
static fir::AliasAnalysis::SourceKind
classifyAllocateFromEffects(OpResult result) {
  std::optional<bool> isNewAllocation = fir::isNewAllocationResult(result);
  return isNewAllocation.value_or(false)
             ? fir::AliasAnalysis::SourceKind::Allocate
             : fir::AliasAnalysis::SourceKind::Unknown;
}

//===----------------------------------------------------------------------===//
// AliasAnalysis: alias
//===----------------------------------------------------------------------===//

static fir::AliasAnalysis::Source::Attributes
getAttrsFromVariable(fir::FortranVariableOpInterface var) {
  fir::AliasAnalysis::Source::Attributes attrs;
  if (var.isTarget())
    attrs.set(fir::AliasAnalysis::Attribute::Target);
  if (var.isPointer())
    attrs.set(fir::AliasAnalysis::Attribute::Pointer);
  if (var.isIntentIn())
    attrs.set(fir::AliasAnalysis::Attribute::IntentIn);
  if (var.isCrayPointer())
    attrs.set(fir::AliasAnalysis::Attribute::CrayPointer);
  if (var.isCrayPointee())
    attrs.set(fir::AliasAnalysis::Attribute::CrayPointee);

  return attrs;
}

bool fir::AliasAnalysis::symbolMayHaveTargetAttr(mlir::SymbolRefAttr symbol,
                                                 mlir::Operation *from) {
  assert(from);

  // If we cannot find the nearest SymbolTable assume the worst.
  const mlir::SymbolTable *symTab = getNearestSymbolTable(from);
  if (!symTab)
    return true;

  if (auto globalOp = symTab->lookup<fir::GlobalOp>(symbol.getLeafReference()))
    return globalOp.getTarget().value_or(false);

  // If the symbol is not defined by fir.global assume the worst.
  return true;
}

static bool isEvaluateInMemoryBlockArg(mlir::Value v) {
  if (auto evalInMem = llvm::dyn_cast_or_null<hlfir::EvaluateInMemoryOp>(
          v.getParentRegion()->getParentOp()))
    return evalInMem.getMemory() == v;
  return false;
}

template <typename OMPTypeOp, typename DeclTypeOp>
static bool isPrivateArg(omp::BlockArgOpenMPOpInterface &argIface,
                         OMPTypeOp &op, DeclTypeOp &declOp) {
  if (!op.getPrivateSyms().has_value())
    return false;
  for (auto [opSym, blockArg] :
       llvm::zip_equal(*op.getPrivateSyms(), argIface.getPrivateBlockArgs())) {
    if (blockArg == declOp.getMemref()) {
      return true;
    }
  }
  return false;
}

/// Classify `mappedValue` when defined by OpenACC mapping op `accOp`.
/// Private-like ops use `SourceKind::Allocate`; other data clauses use
/// `getSourceFn` on the mapped host variable (`mlir::acc::getVar`).
static fir::AliasAnalysis::Source getSourceForACCMappedValue(
    mlir::Value mappedValue, mlir::Operation *accOp,
    llvm::function_ref<fir::AliasAnalysis::Source(mlir::Value)> getSourceFn,
    bool originIsData,
    fir::AliasAnalysis::Source::Attributes accumulatedAttrs) {
  assert(accOp && "OpenACC mapping op required");
  // Private-like ops use SourceKind::Allocate.
  if (mlir::isa<mlir::acc::ReductionInitOp, mlir::acc::PrivateOp,
                mlir::acc::FirstprivateOp, mlir::acc::FirstprivateMapInitialOp>(
          accOp))
    return {{mappedValue, nullptr, originIsData},
            fir::AliasAnalysis::SourceKind::Allocate,
            mappedValue.getType(),
            accumulatedAttrs,
            /*approximateSource=*/false,
            /*accessPath=*/{},
            /*isCapturedInInternalProcedure=*/false};

  // Not private-like: classify using the corresponding host variable's source.
  //
  // Caveat: with discrete device memory, host and device copies do not alias
  // even when this path makes them look related. Alias analysis here is usually
  // about two values *inside* a compute region, not host-vs-device pointer
  // queries, so using the host source remains a reasonable tradeoff for
  // disambiguating in-region uses. Finer modeling would require extending
  // AliasAnalysis::Source (with address space) and teaching AA to use it.
  fir::AliasAnalysis::Source source = getSourceFn(mlir::acc::getVar(accOp));
  source.attributes |= accumulatedAttrs;
  return source;
}

namespace fir {

void AliasAnalysis::Source::AccessPath::print(llvm::raw_ostream &os) const {
  os << "[";
  for (auto it = steps.begin(); it != steps.end(); ++it) {
    if (it != steps.begin())
      os << ", ";
    switch (it->kind) {
    case PathStep::Kind::Component:
      os << "Component(\"" << it->component.getValue() << "\")";
      break;
    case PathStep::Kind::PointerDeref:
      os << "PointerDeref";
      break;
    case PathStep::Kind::AllocDeref:
      os << "AllocDeref";
      break;
    }
  }
  os << "]";
  if (isApproximate)
    os << "(~)";
}

void AliasAnalysis::Source::print(llvm::raw_ostream &os) const {
  if (auto v = llvm::dyn_cast<mlir::Value>(origin.u))
    os << v;
  else if (auto gbl = llvm::dyn_cast<mlir::SymbolRefAttr>(origin.u))
    os << gbl;
  os << " SourceKind: " << EnumToString(kind);
  os << " Type: " << valueType << " ";
  if (origin.isData) {
    os << " following data ";
  } else {
    os << " following box reference ";
  }
  os << " AccessPath: ";
  accessPath.print(os);
  os << " ";
  attributes.Dump(os, EnumToString);
}

bool AliasAnalysis::isRecordWithPointerComponent(mlir::Type ty) {
  auto eleTy = fir::dyn_cast_ptrEleTy(ty);
  if (!eleTy)
    return false;
  // TO DO: Look for pointer components
  return mlir::isa<fir::RecordType>(eleTy);
}

bool AliasAnalysis::isPointerReference(mlir::Type ty) {
  auto eleTy = fir::dyn_cast_ptrEleTy(ty);
  if (!eleTy)
    return false;

  return fir::isPointerType(eleTy) || mlir::isa<fir::PointerType>(eleTy);
}

bool AliasAnalysis::Source::isTargetOrPointer() const {
  return attributes.test(Attribute::Pointer) ||
         attributes.test(Attribute::Target);
}

bool AliasAnalysis::Source::isTarget() const {
  return attributes.test(Attribute::Target);
}

bool AliasAnalysis::Source::isPointer() const {
  return attributes.test(Attribute::Pointer);
}

bool AliasAnalysis::Source::isCrayPointee() const {
  return attributes.test(Attribute::CrayPointee);
}

bool AliasAnalysis::Source::isCrayPointer() const {
  return attributes.test(Attribute::CrayPointer);
}

bool AliasAnalysis::Source::isCrayPointerOrPointee() const {
  return isCrayPointer() || isCrayPointee();
}

bool AliasAnalysis::Source::isDummyArgument() const {
  if (auto v = origin.u.dyn_cast<mlir::Value>()) {
    return fir::isDummyArgument(v);
  }
  return false;
}

bool AliasAnalysis::Source::isData() const { return origin.isData; }
bool AliasAnalysis::Source::isBoxData() const {
  return mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(valueType)) &&
         origin.isData;
}

bool AliasAnalysis::Source::isFortranUserVariable() const {
  if (!origin.instantiationPoint)
    return false;
  return llvm::TypeSwitch<mlir::Operation *, bool>(origin.instantiationPoint)
      .template Case<fir::DeclareOp, hlfir::DeclareOp>([&](auto declOp) {
        return fir::NameUniquer::deconstruct(declOp.getUniqName()).first ==
               fir::NameUniquer::NameKind::VARIABLE;
      })
      .Default([&](auto op) { return false; });
}

bool AliasAnalysis::Source::mayBeDummyArgOrHostAssoc() const {
  return kind != SourceKind::Allocate && kind != SourceKind::Global;
}

bool AliasAnalysis::Source::mayBePtrDummyArgOrHostAssoc() const {
  // Must alias like dummy arg (or HostAssoc).
  if (!mayBeDummyArgOrHostAssoc())
    return false;
  // Must be address of the dummy arg not of a dummy arg component.
  if (isRecordWithPointerComponent(valueType))
    return false;
  // Must be address *of* (not *in*) a pointer.
  return attributes.test(Attribute::Pointer) && !isData();
}

bool AliasAnalysis::Source::mayBeActualArg() const {
  return kind != SourceKind::Allocate;
}

bool AliasAnalysis::Source::mayBeActualArgWithPtr(
    const mlir::Value *val) const {
  // Must not be local.
  if (!mayBeActualArg())
    return false;
  // Can be address *of* (not *in*) a pointer.
  if (attributes.test(Attribute::Pointer) && !isData())
    return true;
  // Can be address of a composite with a pointer component.
  if (isRecordWithPointerComponent(val->getType()))
    return true;
  return false;
}

// Return true if the two locations cannot alias based
// on the access data type, e.g. an address of a descriptor
// cannot alias with an address of data (unless the data
// may contain a descriptor).
static bool noAliasBasedOnType(mlir::Value lhs, mlir::Value rhs) {
  mlir::Type lhsType = lhs.getType();
  mlir::Type rhsType = rhs.getType();
  if (!fir::isa_ref_type(lhsType) || !fir::isa_ref_type(rhsType))
    return false;
  mlir::Type lhsElemType = fir::unwrapRefType(lhsType);
  mlir::Type rhsElemType = fir::unwrapRefType(rhsType);
  if (mlir::isa<fir::BaseBoxType>(lhsElemType) !=
      mlir::isa<fir::BaseBoxType>(rhsElemType)) {
    // One of the types is fir.box and another is not.
    mlir::Type nonBoxType;
    if (mlir::isa<fir::BaseBoxType>(lhsElemType))
      nonBoxType = rhsElemType;
    else
      nonBoxType = lhsElemType;

    if (!fir::isRecordWithDescriptorMember(nonBoxType)) {
      LLVM_DEBUG(llvm::dbgs() << "  no alias based on the access types\n");
      return true;
    }
  }
  return false;
}

/// Return true if two access paths from the same origin variable diverge at
/// a named component step, meaning they address disjoint subobjects of the
/// root variable. For example, paths [Component("a")] and [Component("b")]
/// diverge immediately, while [Component("a"), Component("x")] and
/// [Component("a"), Component("y")] share a common prefix "a" and diverge
/// at the second step.
///
/// When either path continues through a PointerDeref or AllocDeref after
/// the divergence point, the runtime address could potentially reach a
/// sibling subobject only if that sibling is a valid pointer target.
/// A subobject has TARGET when the root variable has the TARGET attribute
/// (Fortran 2018 8.5.7), or when we arrived at the current level through
/// a PointerDeref (the pointer target carries TARGET by definition).
/// When neither condition holds, the pointer cannot be associated with a
/// sibling subobject and the addresses are still disjoint.  Note that the
/// source's POINTER attribute reflects the component traversed during the
/// walk, not the root variable, so we check only TARGET on the source.
///
/// One exception: if BOTH sides end with a PointerDeref, the two pointers
/// could independently be associated with the same third-party TARGET
/// variable, so we conservatively return false.
static bool pathsDivergeAtComponent(const fir::AliasAnalysis::Source &lhsSrc,
                                    const fir::AliasAnalysis::Source &rhsSrc) {
  using PathStep = fir::AliasAnalysis::Source::PathStep;
  auto &lhsSteps = lhsSrc.accessPath.steps;
  auto &rhsSteps = rhsSrc.accessPath.steps;
  size_t minLen = std::min(lhsSteps.size(), rhsSteps.size());
  for (size_t i = 0; i < minLen; ++i) {
    if (lhsSteps[i].kind == PathStep::Kind::Component &&
        rhsSteps[i].kind == PathStep::Kind::Component &&
        lhsSteps[i].component != rhsSteps[i].component) {
      auto hasPtrDerefAfter = [](llvm::ArrayRef<PathStep> steps, size_t from) {
        for (size_t j = from; j < steps.size(); ++j)
          if (steps[j].kind == PathStep::Kind::PointerDeref)
            return true;
        return false;
      };
      bool lhsHasPtrDeref = hasPtrDerefAfter(lhsSteps, i + 1);
      bool rhsHasPtrDeref = hasPtrDerefAfter(rhsSteps, i + 1);
      if (lhsHasPtrDeref && rhsHasPtrDeref)
        return false;
      if (lhsHasPtrDeref || rhsHasPtrDeref) {
        for (size_t j = 0; j < i; ++j)
          if (lhsSteps[j].kind == PathStep::Kind::PointerDeref)
            return false;
        if (lhsSrc.isTarget() || rhsSrc.isTarget())
          return false;
      }
      return true;
    }
    if (lhsSteps[i] != rhsSteps[i])
      break;
  }
  return false;
}

/// Walk backward from \p val through FortranObjectViewOpInterface ops
/// that have zero offset (i.e. they access the same base address).
/// Return the root value at the end of the chain.
static mlir::Value getZeroOffsetViewRoot(mlir::Value val) {
  while (auto *defOp = val.getDefiningOp()) {
    auto viewOp = mlir::dyn_cast<fir::FortranObjectViewOpInterface>(defOp);
    if (!viewOp)
      break;
    auto offset = viewOp.getViewOffset(mlir::cast<mlir::OpResult>(val));
    if (!offset || *offset != 0)
      break;
    val = viewOp.getViewSource(mlir::cast<mlir::OpResult>(val));
  }
  return val;
}

AliasResult AliasAnalysis::alias(mlir::Value lhs, mlir::Value rhs) {
  // A wrapper around alias(Source lhsSrc, Source rhsSrc, mlir::Value lhs,
  // mlir::Value rhs) This allows a user to provide Source that may be obtained
  // through other dialects
  auto lhsSrc = getSource(lhs);
  auto rhsSrc = getSource(rhs);
  return alias(lhsSrc, rhsSrc, lhs, rhs);
}

AliasResult AliasAnalysis::alias(Source lhsSrc, Source rhsSrc, mlir::Value lhs,
                                 mlir::Value rhs) {
  // TODO: alias() has to be aware of the function scopes.
  // After MLIR inlining, the current implementation may
  // not recognize non-aliasing entities.

  // If both values trace back to the same root through zero-offset view
  // operations (e.g. embox without slice, declare, convert), they access
  // the same underlying memory. This check avoids the case where
  // getSource() traces through upstream operations (e.g. a sliced embox)
  // that set approximateSource, conservatively preventing MustAlias.
  if (lhs == rhs || getZeroOffsetViewRoot(lhs) == getZeroOffsetViewRoot(rhs))
    return AliasResult::MustAlias;

  bool approximateSource = lhsSrc.approximateSource || rhsSrc.approximateSource;
  LLVM_DEBUG(llvm::dbgs() << "\nAliasAnalysis::alias\n";
             llvm::dbgs() << "  lhs: " << lhs << "\n";
             llvm::dbgs() << "  lhsSrc: " << lhsSrc << "\n";
             llvm::dbgs() << "  rhs: " << rhs << "\n";
             llvm::dbgs() << "  rhsSrc: " << rhsSrc << "\n";);

  // Disambiguate data and descriptors addresses.
  if (noAliasBasedOnType(lhs, rhs))
    return AliasResult::NoAlias;

  // Indirect case currently not handled. Conservatively assume
  // it aliases with everything
  if (lhsSrc.kind >= SourceKind::Indirect ||
      rhsSrc.kind >= SourceKind::Indirect) {
    LLVM_DEBUG(llvm::dbgs() << "  aliasing because of indirect access\n");
    return AliasResult::MayAlias;
  }

  // After a POINTER dereference the actual address is determined at runtime
  // by pointer association (Fortran 2018 8.5.7, 15.5.2.13). A POINTER can
  // only be associated with a TARGET or another POINTER, so the dereferenced
  // address may alias any source that carries the TARGET or POINTER attribute.
  // When both sides trace to the same origin variable, the pointer deref
  // does not introduce cross-variable aliasing, so this check is skipped
  // (the normal same-origin logic handles that case).
  if (lhsSrc.origin.u != rhsSrc.origin.u &&
      ((lhsSrc.accessPath.hasPointerDeref() && rhsSrc.isTargetOrPointer()) ||
       (rhsSrc.accessPath.hasPointerDeref() && lhsSrc.isTargetOrPointer()))) {
    LLVM_DEBUG(llvm::dbgs()
               << "  aliasing because pointer dereference may reach "
               << "target/pointer\n");
    return AliasResult::MayAlias;
  }

  // Cray pointers/pointees can alias with anything via LOC.
  if (supportCrayPointers) {
    if (lhsSrc.isCrayPointerOrPointee() || rhsSrc.isCrayPointerOrPointee()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  aliasing because of Cray pointer/pointee\n");
      return AliasResult::MayAlias;
    }
  }

  if (lhsSrc.kind == rhsSrc.kind) {
    // If the kinds and origins are the same, then lhs and rhs must alias unless
    // either source is approximate.  Approximate sources are for parts of the
    // origin, but we don't have info here on which parts and whether they
    // overlap, so we normally return MayAlias in that case.
    if (lhsSrc.origin == rhsSrc.origin) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  aliasing because same source kind and origin\n");
      if (approximateSource) {
        if (pathsDivergeAtComponent(lhsSrc, rhsSrc)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  no alias: different components of same origin\n");
          return AliasResult::NoAlias;
        }
        return AliasResult::MayAlias;
      }
      // One should be careful about relying on MustAlias.
      // The LLVM definition implies that the two MustAlias
      // memory objects start at exactly the same location.
      // With Fortran array slices two objects may have
      // the same starting location, but otherwise represent
      // partially overlapping memory locations, e.g.:
      //   integer :: a(10)
      //   ... a(5:1:-1) ! starts at a(5) and addresses a(5), ..., a(1)
      //   ... a(5:10:1) ! starts at a(5) and addresses a(5), ..., a(10)
      // The current implementation of FIR alias analysis will always
      // return MayAlias for such cases.
      return AliasResult::MustAlias;
    }
    // If one value is the address of a composite, and if the other value is the
    // address of a pointer/allocatable component of that composite, their
    // origins compare unequal because the latter has !isData().  As for the
    // address of any component vs. the address of the composite, a store to one
    // can affect a load from the other, so the result should be MayAlias.  To
    // catch this case, we conservatively return MayAlias when one value is the
    // address of a composite, the other value is non-data, and they have the
    // same origin value.
    //
    // TODO: That logic does not check that the latter is actually a component
    // of the former, so it can return MayAlias when unnecessary.  For example,
    // they might both be addresses of components of a larger composite.
    //
    // FIXME: Actually, we should generalize from isRecordWithPointerComponent
    // to any composite because a component with !isData() is not always a
    // pointer.  However, Source::isRecordWithPointerComponent currently doesn't
    // actually check for pointer components, so it's fine for now.
    if (lhsSrc.origin.u == rhsSrc.origin.u &&
        ((isRecordWithPointerComponent(lhs.getType()) && !rhsSrc.isData()) ||
         (isRecordWithPointerComponent(rhs.getType()) && !lhsSrc.isData()))) {
      if (pathsDivergeAtComponent(lhsSrc, rhsSrc)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  no alias: different components of same origin\n");
        return AliasResult::NoAlias;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "  aliasing between composite and non-data component with "
                 << "same source kind and origin value\n");
      return AliasResult::MayAlias;
    }

    // Two host associated accesses may overlap due to an equivalence.
    if (lhsSrc.kind == SourceKind::HostAssoc) {
      LLVM_DEBUG(llvm::dbgs() << "  aliasing because of host association\n");
      return AliasResult::MayAlias;
    }
  }

  Source *src1, *src2;
  mlir::Value *val1, *val2;
  if (lhsSrc.kind < rhsSrc.kind) {
    src1 = &lhsSrc;
    src2 = &rhsSrc;
    val1 = &lhs;
    val2 = &rhs;
  } else {
    src1 = &rhsSrc;
    src2 = &lhsSrc;
    val1 = &rhs;
    val2 = &lhs;
  }

  if (src1->kind == SourceKind::Argument &&
      src2->kind == SourceKind::HostAssoc) {
    // Treat the host entity as TARGET for the purpose of disambiguating
    // it with a dummy access. It is required for this particular case:
    // subroutine test
    //   integer :: x(10)
    //   call inner(x)
    // contains
    //   subroutine inner(y)
    //     integer, target :: y(:)
    //     x(1) = y(1)
    //   end subroutine inner
    // end subroutine test
    //
    // F18 15.5.2.13 (4) (b) allows 'x' and 'y' to address the same object.
    // 'y' has an explicit TARGET attribute, but 'x' has neither TARGET
    // nor POINTER.
    src2->attributes.set(Attribute::Target);
  }

  // Two TARGET/POINTERs may alias.  The logic here focuses on data.  Handling
  // of non-data is included below.
  if (src1->isTargetOrPointer() && src2->isTargetOrPointer() &&
      src1->isData() && src2->isData()) {
    // Two distinct TARGET globals may not alias.
    if (!src1->isPointer() && !src2->isPointer() &&
        src1->kind == SourceKind::Global && src2->kind == SourceKind::Global &&
        src1->origin.u != src2->origin.u) {
      return AliasResult::NoAlias;
    }
    LLVM_DEBUG(llvm::dbgs() << "  aliasing because of target or pointer\n");
    return AliasResult::MayAlias;
  }

  // Aliasing for dummy arg with target attribute.
  //
  // The address of a dummy arg (or HostAssoc) may alias the address of a
  // non-local (global or another dummy arg) when both have target attributes.
  // If either is a composite, addresses of components may alias as well.
  //
  // The previous "if" calling isTargetOrPointer casts a very wide net and so
  // reports MayAlias for many such cases that would otherwise be reported here.
  // It specifically skips such cases where one or both values have !isData()
  // (e.g., address *of* pointer/allocatable component vs. address of
  // composite), so this "if" catches those cases.
  if (src1->attributes.test(Attribute::Target) &&
      src2->attributes.test(Attribute::Target) &&
      ((src1->mayBeDummyArgOrHostAssoc() && src2->mayBeActualArg()) ||
       (src2->mayBeDummyArgOrHostAssoc() && src1->mayBeActualArg()))) {
    LLVM_DEBUG(llvm::dbgs()
               << "  aliasing between targets where one is a dummy arg\n");
    return AliasResult::MayAlias;
  }

  // Aliasing for dummy arg that is a pointer.
  //
  // The address of a pointer dummy arg (but not a pointer component of a dummy
  // arg) may alias the address of either (1) a non-local pointer or (2) thus a
  // non-local composite with a pointer component.  A non-local might be a
  // global or another dummy arg.  The following is an example of the global
  // composite case:
  //
  // module m
  //   type t
  //      real, pointer :: p
  //   end type
  //   type(t) :: a
  //   type(t) :: b
  // contains
  //   subroutine test(p)
  //     real, pointer :: p
  //     p = 42
  //     a = b
  //     print *, p
  //   end subroutine
  // end module
  // program main
  //   use m
  //   real, target :: x1 = 1
  //   real, target :: x2 = 2
  //   a%p => x1
  //   b%p => x2
  //   call test(a%p)
  // end
  //
  // The dummy argument p is an alias for a%p, even for the purposes of pointer
  // association during the assignment a = b.  Thus, the program should print 2.
  //
  // The same is true when p is HostAssoc.  For example, we might replace the
  // test subroutine above with:
  //
  // subroutine test(p)
  //   real, pointer :: p
  //   call internal()
  // contains
  //   subroutine internal()
  //     p = 42
  //     a = b
  //     print *, p
  //   end subroutine
  // end subroutine
  if ((src1->mayBePtrDummyArgOrHostAssoc() &&
       src2->mayBeActualArgWithPtr(val2)) ||
      (src2->mayBePtrDummyArgOrHostAssoc() &&
       src1->mayBeActualArgWithPtr(val1))) {
    LLVM_DEBUG(llvm::dbgs()
               << "  aliasing between pointer dummy arg and either pointer or "
               << "composite with pointer component\n");
    return AliasResult::MayAlias;
  }

  return AliasResult::NoAlias;
}

//===----------------------------------------------------------------------===//
// AliasAnalysis: getModRef
//===----------------------------------------------------------------------===//

static bool isSavedLocal(const fir::AliasAnalysis::Source &src) {
  if (auto symRef = llvm::dyn_cast<mlir::SymbolRefAttr>(src.origin.u)) {
    auto [nameKind, deconstruct] =
        fir::NameUniquer::deconstruct(symRef.getLeafReference().getValue());
    return nameKind == fir::NameUniquer::NameKind::VARIABLE &&
           !deconstruct.procs.empty();
  }
  return false;
}

bool AliasAnalysis::isCallToFortranUserProcedure(Operation *op) {
  fir::CallOp call = dyn_cast<fir::CallOp>(op);
  if (!call)
    return false;

  // TODO: indirect calls are excluded by these checks. Maybe some attribute is
  // needed to flag user calls in this case.
  if (fir::hasBindcAttr(call))
    return true;
  if (std::optional<SymbolRefAttr> callee = call.getCallee()) {
    if (fir::NameUniquer::deconstruct(callee->getLeafReference().getValue())
            .first == fir::NameUniquer::NameKind::PROCEDURE)
      return true;

    const SymbolTable *symTab = getNearestSymbolTable(call);
    if (!symTab)
      return false;

    if (auto funcOp =
            symTab->lookup<FunctionOpInterface>(callee->getLeafReference()))
      if (auto name = funcOp->getAttrOfType<StringAttr>(
              fir::getInternalFuncNameAttrName()))
        if (fir::NameUniquer::deconstruct(name.getValue()).first ==
            fir::NameUniquer::NameKind::PROCEDURE)
          return true;
  }
  return false;
}

ModRefResult AliasAnalysis::getCallModRef(Operation *op, Value var) {
  auto call = dyn_cast<fir::CallOp>(op);
  if (!call)
    return ModRefResult::getModAndRef();

  // TODO: limit to Fortran functions??
  // 1. Detect variables that can be accessed indirectly.
  fir::AliasAnalysis aliasAnalysis;
  fir::AliasAnalysis::Source varSrc =
      aliasAnalysis.getSource(var, /*getLastInstantiationPoint=*/true);
  // If the variable is not a user variable, we cannot safely assume that
  // Fortran semantics apply (e.g., a bare alloca/allocmem result may very well
  // be placed in an allocatable/pointer descriptor and escape).

  // All the logic below is based on Fortran semantics and only holds if this
  // is a call to a procedure from the Fortran source and this is a variable
  // from the Fortran source. Compiler generated temporaries or functions may
  // not adhere to this semantic.
  // TODO: add some opt-in or op-out mechanism for compiler generated temps.
  // An example of something currently problematic is the allocmem generated for
  // ALLOCATE of allocatable target. It currently does not have the target
  // attribute, which would lead this analysis to believe it cannot escape.
  if (!varSrc.isFortranUserVariable() || !isCallToFortranUserProcedure(call))
    return ModRefResult::getModAndRef();
  // Pointer and target may have been captured.
  if (varSrc.isTargetOrPointer())
    return ModRefResult::getModAndRef();
  // Host associated variables may be addressed indirectly via an internal
  // function call, whether the call is in the parent or an internal procedure.
  // Note that the host associated/internal procedure may be referenced
  // indirectly inside calls to non internal procedure. This is because internal
  // procedures may be captured or passed. As this is tricky to analyze, always
  // consider such variables may be accessed in any calls.
  if (varSrc.kind == fir::AliasAnalysis::SourceKind::HostAssoc ||
      varSrc.isCapturedInInternalProcedure)
    return ModRefResult::getModAndRef();
  // At that stage, it has been ruled out that local (including the saved ones)
  // and dummy cannot be indirectly accessed in the call.
  if (varSrc.kind != fir::AliasAnalysis::SourceKind::Allocate &&
      varSrc.kind != fir::AliasAnalysis::SourceKind::Argument &&
      !varSrc.isDummyArgument()) {
    if (varSrc.kind != fir::AliasAnalysis::SourceKind::Global ||
        !isSavedLocal(varSrc))
      return ModRefResult::getModAndRef();
  }
  // 2. Check if the variable is passed via the arguments.
  for (auto arg : call.getArgs()) {
    if (fir::conformsWithPassByRef(arg.getType()) &&
        !aliasAnalysis.alias(arg, var).isNo()) {
      // TODO: intent(in) would allow returning Ref here. This can be obtained
      // in the func.func attributes for direct calls, but the module lookup is
      // linear with the number of MLIR symbols, which would introduce a pseudo
      // quadratic behavior num_calls * num_func.
      return ModRefResult::getModAndRef();
    }
  }
  // The call cannot access the variable.
  return ModRefResult::getNoModRef();
}

/// This is mostly inspired by MLIR::LocalAliasAnalysis, except that
/// fir.call's are handled in a special way.
ModRefResult AliasAnalysis::getModRef(Operation *op, Value location) {
  if (auto call = llvm::dyn_cast<fir::CallOp>(op)) {
    ModRefResult result = getCallModRef(call, location);
    if (result != ModRefResult::getModAndRef())
      return result;
    // Proceed to MemoryEffectOpInterface analysis in case one
    // is attached for fir.call.
  }

  // Build a ModRefResult by merging the behavior of the effects of this
  // operation.
  ModRefResult result = ModRefResult::getNoModRef();
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (op->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
    for (mlir::Region &region : op->getRegions()) {
      result = result.merge(getModRef(region, location));
      if (result.isModAndRef())
        break;
    }

    // In MLIR, RecursiveMemoryEffects can be combined with
    // MemoryEffectOpInterface to describe extra effects on top of the
    // effects of the nested operations.  However, the presence of
    // RecursiveMemoryEffects and the absence of MemoryEffectOpInterface
    // implies the operation has no other memory effects than the one of its
    // nested operations.
    if (!interface)
      return result;
  }

  if (!interface || result.isModAndRef())
    return ModRefResult::getModAndRef();

  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);

  for (const MemoryEffects::EffectInstance &effect : effects) {
    // MemAlloc and MemFree are not mod-ref effects.
    if (isa<MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect()))
      continue;

    // An effect on a non-addressable resource cannot affect
    // memory pointed to by 'location'.
    mlir::SideEffects::Resource *resource = effect.getResource();
    if (!resource->isAddressable())
      continue;

    // Check for an alias between the effect and our memory location.
    AliasResult aliasResult = AliasResult::MayAlias;
    if (Value effectValue = effect.getValue())
      aliasResult = alias(effectValue, location);

    // If we don't alias, ignore this effect.
    if (aliasResult.isNo())
      continue;

    // Merge in the corresponding mod or ref for this effect.
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      result = result.merge(ModRefResult::getRef());
    else
      result = result.merge(ModRefResult::getMod());

    if (result.isModAndRef())
      break;
  }
  return result;
}

ModRefResult AliasAnalysis::getModRef(mlir::Region &region,
                                      mlir::Value location) {
  ModRefResult result = ModRefResult::getNoModRef();
  for (mlir::Operation &op : region.getOps()) {
    result = result.merge(getModRef(&op, location));
    if (result.isModAndRef())
      return result;
  }
  return result;
}

AliasAnalysis::Source AliasAnalysis::getSource(mlir::Value v,
                                               bool getLastInstantiationPoint) {
  auto *defOp = v.getDefiningOp();
  SourceKind type{SourceKind::Unknown};
  mlir::Type ty;
  bool breakFromLoop{false};
  bool approximateSource{false};
  bool isCapturedInInternalProcedure{false};
  bool followBoxData{mlir::isa<fir::BaseBoxType>(v.getType())};
  bool isBoxRef{fir::isa_ref_type(v.getType()) &&
                mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(v.getType()))};
  bool followingData = !isBoxRef;
  mlir::SymbolRefAttr global;
  Source::Attributes attributes;
  mlir::Operation *instantiationPoint{nullptr};

  // Access path steps collected during the backward walk (leaf-to-root order).
  // Reversed into the final AccessPath at the end, unless the box-load branch
  // composes the full path directly.
  llvm::SmallVector<Source::PathStep, 4> pathSteps;
  Source::AccessPath accessPath;
  bool accessPathFinalized{false};
  while (defOp && !breakFromLoop) {
    // Operations may have multiple results, so we need to analyze
    // the result for which the source is queried.
    auto opResult = mlir::cast<OpResult>(v);
    assert(opResult.getOwner() == defOp && "v must be a result of defOp");
    // Value-scoped allocation detection via effects.
    if (classifyAllocateFromEffects(opResult) == SourceKind::Allocate) {
      type = SourceKind::Allocate;
      break;
    }
    ty = opResult.getType();
    std::optional<AliasAnalysis::Source> accSourceReturn;
    llvm::TypeSwitch<Operation *>(defOp)
        .Case([&](hlfir::AsExprOp op) {
          // TODO: we should probably always report hlfir.as_expr
          // as a unique source, and let the codegen decide whether
          // to use the original buffer or create a copy.
          v = op.getVar();
          defOp = v.getDefiningOp();
        })
        .Case([&](hlfir::AssociateOp op) {
          assert(opResult != op.getMustFreeStrorageFlag() &&
                 "MustFreeStorageFlag result is not an aliasing candidate");

          mlir::Value source = op.getSource();
          if (fir::isa_trivial(source.getType())) {
            // Trivial values will always use distinct temp memory,
            // so we can classify this as Allocate and stop.
            type = SourceKind::Allocate;
            breakFromLoop = true;
          } else {
            // AssociateOp may reuse the expression storage,
            // so we have to trace further.
            v = source;
            defOp = v.getDefiningOp();
          }
        })
        .Case([&](fir::PackArrayOp op) {
          // The packed array is not distinguishable from the original
          // array, so skip PackArrayOp and track further through
          // the array operand.
          v = op.getArray();
          defOp = v.getDefiningOp();
          approximateSource = true;
        })
        .Case([&](fir::AbsentOp op) {
          // Although fir.absent is not a local allocation, we treat it
          // similarly so that it can be disambiguated that it doesn't alias any
          // other values. Two entities coming from separate fir.absent ops
          // also do not alias each other.
          type = SourceKind::Allocate;
          breakFromLoop = true;
        })
        .Case([&](fir::LoadOp op) {
          // If load is inside target and it points to mapped item,
          // continue tracking.
          Operation *loadMemrefOp = op.getMemref().getDefiningOp();
          bool isDeclareOp =
              llvm::isa_and_present<fir::DeclareOp>(loadMemrefOp) ||
              llvm::isa_and_present<hlfir::DeclareOp>(loadMemrefOp);
          if (isDeclareOp &&
              llvm::isa<omp::TargetOp>(loadMemrefOp->getParentOp())) {
            v = op.getMemref();
            defOp = v.getDefiningOp();
            return;
          }

          // Loading a box value from memory (e.g. a pointer/allocatable
          // component's descriptor). Trace the memref so derived-type
          // component accesses reach their [hl]fir.declare instead of
          // SourceKind::Indirect (which forces MayAlias broadly in alias()).
          // The access path records a PointerDeref or AllocDeref step here
          // so that alias() can distinguish pointer-dereferenced addresses
          // from statically known ones.
          if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty); boxTy) {

            bool isPointerBox = mlir::isa<fir::PointerType>(boxTy.getEleTy());
            if (isPointerBox)
              attributes.set(Attribute::Pointer);

            auto boxSrc = getSource(op.getMemref());
            attributes |= boxSrc.attributes;
            approximateSource |= boxSrc.approximateSource;
            isCapturedInInternalProcedure |=
                boxSrc.isCapturedInInternalProcedure;

            if (getLastInstantiationPoint) {
              if (!instantiationPoint)
                instantiationPoint = boxSrc.origin.instantiationPoint;
            } else {
              instantiationPoint = boxSrc.origin.instantiationPoint;
            }

            // Compose the access path: inner path (root to this load point)
            // + deref step + outer path (this load to the queried value).
            accessPath.steps = boxSrc.accessPath.steps;
            Source::PathStep derefStep;
            derefStep.kind = isPointerBox ? Source::PathStep::Kind::PointerDeref
                                          : Source::PathStep::Kind::AllocDeref;
            derefStep.component = {};
            accessPath.steps.push_back(derefStep);
            for (int i = pathSteps.size() - 1; i >= 0; --i)
              accessPath.steps.push_back(pathSteps[i]);
            accessPath.isApproximate =
                boxSrc.accessPath.isApproximate || approximateSource;
            accessPathFinalized = true;

            global = llvm::dyn_cast<mlir::SymbolRefAttr>(boxSrc.origin.u);
            if (global) {
              type = SourceKind::Global;
            } else {
              auto def = llvm::cast<mlir::Value>(boxSrc.origin.u);
              bool classified = false;
              if (auto defAsOpResult = mlir::dyn_cast<OpResult>(def)) {
                if (classifyAllocateFromEffects(defAsOpResult) ==
                    SourceKind::Allocate) {
                  v = def;
                  defOp = defAsOpResult.getOwner();
                  type = SourceKind::Allocate;
                  classified = true;
                }
              }
              if (!classified) {
                if (boxSrc.kind == SourceKind::Allocate) {
                  type = SourceKind::Allocate;
                  v = def;
                  defOp = nullptr;
                } else if (isDummyArgument(def)) {
                  defOp = nullptr;
                  v = def;
                } else {
                  type = SourceKind::Indirect;
                }
              }
            }
            breakFromLoop = true;
            return;
          }
          // No further tracking for addresses loaded from memory for now.
          type = SourceKind::Indirect;
          breakFromLoop = true;
        })
        .Case<fir::AddrOfOp, cuf::DeviceAddressOp>([&](auto op) {
          // Address of a global scope object.
          ty = v.getType();
          type = SourceKind::Global;
          // TODO: Take followBoxData into account when setting the pointer
          // attribute
          if (isPointerReference(ty))
            attributes.set(Attribute::Pointer);

          if constexpr (std::is_same_v<std::decay_t<decltype(op)>,
                                       fir::AddrOfOp>)
            global = op.getSymbol();
          else if constexpr (std::is_same_v<std::decay_t<decltype(op)>,
                                            cuf::DeviceAddressOp>)
            global = op.getHostSymbol();
          else
            llvm_unreachable("unexpected operation");

          if (symbolMayHaveTargetAttr(global, op))
            attributes.set(Attribute::Target);

          breakFromLoop = true;
        })
        .Case<hlfir::DeclareOp, fir::DeclareOp>([&](auto op) {
          // The declare operations support FortranObjectViewOpInterface,
          // but their handling is more complex. Maybe we can find better
          // abstractions to handle them in a general fashion.
          bool isPrivateItem = false;
          if (omp::BlockArgOpenMPOpInterface argIface =
                  dyn_cast<omp::BlockArgOpenMPOpInterface>(op->getParentOp())) {
            Value ompValArg;
            llvm::TypeSwitch<Operation *>(op->getParentOp())
                .Case([&](omp::TargetOp targetOp) {
                  // If declare operation is inside omp target region,
                  // continue alias analysis outside the target region
                  for (auto [opArg, blockArg] : llvm::zip_equal(
                           targetOp.getMapVars(), argIface.getMapBlockArgs())) {
                    if (blockArg == op.getMemref()) {
                      omp::MapInfoOp mapInfo =
                          llvm::cast<omp::MapInfoOp>(opArg.getDefiningOp());
                      ompValArg = mapInfo.getVarPtr();
                      return;
                    }
                  }
                  // If given operation does not reflect mapping item,
                  // check private clause
                  isPrivateItem = isPrivateArg(argIface, targetOp, op);
                })
                .template Case<omp::DistributeOp, omp::ParallelOp,
                               omp::SectionsOp, omp::SimdOp, omp::SingleOp,
                               omp::TaskloopContextOp, omp::TaskOp,
                               omp::WsloopOp>([&](auto privateOp) {
                  isPrivateItem = isPrivateArg(argIface, privateOp, op);
                });
            if (ompValArg) {
              v = ompValArg;
              defOp = ompValArg.getDefiningOp();
              return;
            }
          }
          auto varIf = llvm::cast<fir::FortranVariableOpInterface>(defOp);
          // While going through a declare operation collect
          // the variable attributes from it. Right now, some
          // of the attributes are duplicated, e.g. a TARGET dummy
          // argument has the target attribute both on its declare
          // operation and on the entry block argument.
          // In case of host associated use, the declare operation
          // is the only carrier of the variable attributes,
          // so we have to collect them here.
          attributes |= getAttrsFromVariable(varIf);
          isCapturedInInternalProcedure |=
              varIf.isCapturedInInternalProcedure();
          if (varIf.isHostAssoc()) {
            // Do not track past such DeclareOp, because it does not
            // currently provide any useful information. The host associated
            // access will end up dereferencing the host association tuple,
            // so we may as well stop right now.
            v = opResult;
            // TODO: if the host associated variable is a dummy argument
            // of the host, I think, we can treat it as SourceKind::Argument
            // for the purpose of alias analysis inside the internal procedure.
            type = SourceKind::HostAssoc;
            breakFromLoop = true;
            return;
          }
          if (getLastInstantiationPoint) {
            // Fetch only the innermost instantiation point.
            if (!instantiationPoint)
              instantiationPoint = op;

            if (op.getDummyScope()) {
              // Do not track past DeclareOp that has the dummy_scope
              // operand. This DeclareOp is known to represent
              // a dummy argument for some runtime instantiation
              // of a procedure.
              type = SourceKind::Argument;
              breakFromLoop = true;
              return;
            }
          } else {
            instantiationPoint = op;
          }
          if (isPrivateItem) {
            type = SourceKind::Allocate;
            breakFromLoop = true;
            return;
          }
          // TODO: Look for the fortran attributes present on the operation
          // Track further through the operand
          v = op.getMemref();
          defOp = v.getDefiningOp();
        })
        .Case([&](fir::FortranObjectViewOpInterface op) {
          // This case must be located after the cases for concrete
          // operations that support FortraObjectViewOpInterface,
          // so that their special handling kicks in.

          // fir.embox/rebox case: this is the only case where we check
          // for followBoxData.
          // TODO: it looks like we do not have LIT tests that fail
          // upon removal of the followBoxData code. We should come up
          // with a test or remove this code.
          if (!followBoxData &&
              (mlir::isa<fir::EmboxOp>(op) || mlir::isa<fir::ReboxOp>(op))) {
            breakFromLoop = true;
            return;
          }

          // Record component access steps for the access path.
          //
          // hlfir.designate carries the component name directly as a
          // StringAttr, e.g. hlfir.designate %x{"fieldName"}.
          if (auto designateOp = mlir::dyn_cast<hlfir::DesignateOp>(defOp)) {
            if (auto comp = designateOp.getComponent()) {
              Source::PathStep step;
              step.kind = Source::PathStep::Kind::Component;
              step.component = *comp;
              pathSteps.push_back(step);
            }
          } else if (auto coordOp = mlir::dyn_cast<fir::CoordinateOp>(defOp)) {
            // fir.coordinate_of encodes field accesses as integer indices
            // into the record type's field list (the field_indices attr).
            // A single coordinate_of may access multiple nested fields,
            // e.g. fir.coordinate_of %obj, inner, a has field_indices
            // [inner_idx, a_idx].  Walk the type hierarchy to recover
            // the field name for each static index.  Dynamic indices
            // (kDynamicIndex) correspond to array subscripts, not named
            // components, so they only advance the type through the
            // array dimension.
            std::optional<llvm::ArrayRef<int32_t>> fieldIndices =
                coordOp.getFieldIndices();
            if (fieldIndices) {
              mlir::Type currentTy =
                  fir::dyn_cast_ptrOrBoxEleTy(coordOp.getRef().getType());
              llvm::SmallVector<mlir::StringAttr, 4> componentNames;
              unsigned dimension = 0;
              for (int32_t idx : *fieldIndices) {
                if (idx == fir::CoordinateOp::kDynamicIndex) {
                  if (dimension == 0) {
                    if (auto seqTy =
                            mlir::dyn_cast<fir::SequenceType>(currentTy))
                      dimension = seqTy.getDimension();
                  }
                  if (dimension) {
                    if (--dimension == 0)
                      currentTy = mlir::cast<fir::SequenceType>(currentTy)
                                      .getElementType();
                  }
                  continue;
                }
                auto recTy = mlir::dyn_cast<fir::RecordType>(currentTy);
                if (!recTy) {
                  // Unexpected type structure; discard any partially
                  // collected names so the access path stays conservative
                  // rather than recording a misleading partial path.
                  componentNames.clear();
                  break;
                }
                auto typeList = recTy.getTypeList();
                if (idx < 0 || static_cast<size_t>(idx) >= typeList.size()) {
                  // Out-of-bounds field index; same conservative treatment.
                  componentNames.clear();
                  break;
                }
                componentNames.push_back(mlir::StringAttr::get(
                    defOp->getContext(), typeList[idx].first));
                currentTy = typeList[idx].second;
              }
              // pathSteps is in leaf-to-root order (reversed at the end),
              // so push innermost component first.
              for (auto it = componentNames.rbegin();
                   it != componentNames.rend(); ++it) {
                Source::PathStep step;
                step.kind = Source::PathStep::Kind::Component;
                step.component = *it;
                pathSteps.push_back(step);
              }
            }
          }

          // Collect attributes from FortranVariableOpInterface operations.
          if (auto varIf =
                  mlir::dyn_cast<fir::FortranVariableOpInterface>(defOp))
            attributes |= getAttrsFromVariable(varIf);
          // Set Pointer attribute based on the reference type.
          if (isPointerReference(ty))
            attributes.set(Attribute::Pointer);

          // Update v to point to the operand that represents the object
          // referenced by the operation's result.
          v = op.getViewSource(opResult);
          defOp = v.getDefiningOp();
          // If the input the resulting object references are offsetted,
          // then set approximateSource.
          auto offset = op.getViewOffset(opResult);
          if (!offset || *offset != 0)
            approximateSource = true;

          // If the source is a box, and the result is not a box,
          // then this is one of the box "unpacking" operations,
          // so we should set followBoxData.
          if (mlir::isa<fir::BaseBoxType>(v.getType()) &&
              !mlir::isa<fir::BaseBoxType>(ty))
            followBoxData = true;
        })
        .Case<ACC_DATA_ENTRY_AND_INIT_OPS>([&](auto op) {
          accSourceReturn = getSourceForACCMappedValue(
              v, op.getOperation(),
              [&](mlir::Value x) {
                return getSource(x, getLastInstantiationPoint);
              },
              followingData, attributes);
          breakFromLoop = true;
        })
        .Default([&](auto op) {
          defOp = nullptr;
          breakFromLoop = true;
        });
    if (accSourceReturn)
      return *accSourceReturn;
  }
  if (!defOp && type == SourceKind::Unknown) {
    // Check if the memory source is coming through a dummy argument.
    if (isDummyArgument(v)) {
      type = SourceKind::Argument;
      ty = v.getType();
      if (fir::valueHasFirAttribute(v, fir::getTargetAttrName()))
        attributes.set(Attribute::Target);

      if (isPointerReference(ty))
        attributes.set(Attribute::Pointer);
    } else if (isEvaluateInMemoryBlockArg(v)) {
      // hlfir.eval_in_mem block operands is allocated by the operation.
      type = SourceKind::Allocate;
      ty = v.getType();
    } else if (mlir::Operation *accOp =
                   mlir::acc::getACCDataClauseOpForBlockArg(v)) {
      return getSourceForACCMappedValue(
          v, accOp,
          [&](mlir::Value x) {
            return getSource(x, getLastInstantiationPoint);
          },
          followingData, attributes);
    }
  }

  // Finalize the access path if not already done by the box-load branch.
  if (!accessPathFinalized) {
    std::reverse(pathSteps.begin(), pathSteps.end());
    accessPath.steps = std::move(pathSteps);
    accessPath.isApproximate = approximateSource;
  }

  if (type == SourceKind::Global) {
    return {{global, instantiationPoint, followingData},
            type,
            ty,
            attributes,
            approximateSource,
            accessPath,
            isCapturedInInternalProcedure};
  }
  return {{v, instantiationPoint, followingData},
          type,
          ty,
          attributes,
          approximateSource,
          accessPath,
          isCapturedInInternalProcedure};
}

const mlir::SymbolTable *
fir::AliasAnalysis::getNearestSymbolTable(mlir::Operation *from) {
  assert(from);
  Operation *symTabOp = mlir::SymbolTable::getNearestSymbolTable(from);
  if (!symTabOp)
    return nullptr;
  auto it = symTabMap.find(symTabOp);
  if (it != symTabMap.end())
    return &it->second;
  return &symTabMap.try_emplace(symTabOp, symTabOp).first->second;
}

} // namespace fir
