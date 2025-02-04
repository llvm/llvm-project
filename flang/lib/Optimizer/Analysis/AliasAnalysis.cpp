//===- AliasAnalysis.cpp - Alias Analysis for FIR  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "fir-alias-analysis"

//===----------------------------------------------------------------------===//
// AliasAnalysis: alias
//===----------------------------------------------------------------------===//

/// Temporary function to skip through all the no op operations
/// TODO: Generalize support of fir.load
static mlir::Value getOriginalDef(mlir::Value v) {
  mlir::Operation *defOp;
  bool breakFromLoop = false;
  while (!breakFromLoop && (defOp = v.getDefiningOp())) {
    llvm::TypeSwitch<Operation *>(defOp)
        .Case<fir::ConvertOp>([&](fir::ConvertOp op) { v = op.getValue(); })
        .Case<fir::DeclareOp, hlfir::DeclareOp>(
            [&](auto op) { v = op.getMemref(); })
        .Default([&](auto op) { breakFromLoop = true; });
  }
  return v;
}

namespace fir {

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

bool AliasAnalysis::Source::isDummyArgument() const {
  if (auto v = origin.u.dyn_cast<mlir::Value>()) {
    return fir::isDummyArgument(v);
  }
  return false;
}

static bool isEvaluateInMemoryBlockArg(mlir::Value v) {
  if (auto evalInMem = llvm::dyn_cast_or_null<hlfir::EvaluateInMemoryOp>(
          v.getParentRegion()->getParentOp()))
    return evalInMem.getMemory() == v;
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
  bool approximateSource = lhsSrc.approximateSource || rhsSrc.approximateSource;
  LLVM_DEBUG(llvm::dbgs() << "\nAliasAnalysis::alias\n";
             llvm::dbgs() << "  lhs: " << lhs << "\n";
             llvm::dbgs() << "  lhsSrc: " << lhsSrc << "\n";
             llvm::dbgs() << "  rhs: " << rhs << "\n";
             llvm::dbgs() << "  rhsSrc: " << rhsSrc << "\n";);

  // Indirect case currently not handled. Conservatively assume
  // it aliases with everything
  if (lhsSrc.kind >= SourceKind::Indirect ||
      rhsSrc.kind >= SourceKind::Indirect) {
    LLVM_DEBUG(llvm::dbgs() << "  aliasing because of indirect access\n");
    return AliasResult::MayAlias;
  }

  if (lhsSrc.kind == rhsSrc.kind) {
    // If the kinds and origins are the same, then lhs and rhs must alias unless
    // either source is approximate.  Approximate sources are for parts of the
    // origin, but we don't have info here on which parts and whether they
    // overlap, so we normally return MayAlias in that case.
    if (lhsSrc.origin == rhsSrc.origin) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  aliasing because same source kind and origin\n");
      if (approximateSource)
        return AliasResult::MayAlias;
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

static bool isCallToFortranUserProcedure(fir::CallOp call) {
  // TODO: indirect calls are excluded by these checks. Maybe some attribute is
  // needed to flag user calls in this case.
  if (fir::hasBindcAttr(call))
    return true;
  if (std::optional<mlir::SymbolRefAttr> callee = call.getCallee())
    return fir::NameUniquer::deconstruct(callee->getLeafReference().getValue())
               .first == fir::NameUniquer::NameKind::PROCEDURE;
  return false;
}

static ModRefResult getCallModRef(fir::CallOp call, mlir::Value var) {
  // TODO: limit to Fortran functions??
  // 1. Detect variables that can be accessed indirectly.
  fir::AliasAnalysis aliasAnalysis;
  fir::AliasAnalysis::Source varSrc = aliasAnalysis.getSource(var);
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

/// This is mostly inspired by MLIR::LocalAliasAnalysis with 2 notable
/// differences 1) Regions are not handled here but will be handled by a data
/// flow analysis to come 2) Allocate and Free effects are considered
/// modifying
ModRefResult AliasAnalysis::getModRef(Operation *op, Value location) {
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    if (auto call = llvm::dyn_cast<fir::CallOp>(op))
      return getCallModRef(call, location);
    return ModRefResult::getModAndRef();
  }

  // Build a ModRefResult by merging the behavior of the effects of this
  // operation.
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);

  ModRefResult result = ModRefResult::getNoModRef();
  for (const MemoryEffects::EffectInstance &effect : effects) {

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
    if (op.hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
      for (mlir::Region &subRegion : op.getRegions()) {
        result = result.merge(getModRef(subRegion, location));
        // Fast return is already mod and ref.
        if (result.isModAndRef())
          return result;
      }
      // In MLIR, RecursiveMemoryEffects can be combined with
      // MemoryEffectOpInterface to describe extra effects on top of the
      // effects of the nested operations.  However, the presence of
      // RecursiveMemoryEffects and the absence of MemoryEffectOpInterface
      // implies the operation has no other memory effects than the one of its
      // nested operations.
      if (!mlir::isa<mlir::MemoryEffectOpInterface>(op))
        continue;
    }
    result = result.merge(getModRef(&op, location));
    if (result.isModAndRef())
      return result;
  }
  return result;
}

AliasAnalysis::Source::Attributes
getAttrsFromVariable(fir::FortranVariableOpInterface var) {
  AliasAnalysis::Source::Attributes attrs;
  if (var.isTarget())
    attrs.set(AliasAnalysis::Attribute::Target);
  if (var.isPointer())
    attrs.set(AliasAnalysis::Attribute::Pointer);
  if (var.isIntentIn())
    attrs.set(AliasAnalysis::Attribute::IntentIn);

  return attrs;
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
  while (defOp && !breakFromLoop) {
    ty = defOp->getResultTypes()[0];
    llvm::TypeSwitch<Operation *>(defOp)
        .Case<hlfir::AsExprOp>([&](auto op) {
          v = op.getVar();
          defOp = v.getDefiningOp();
        })
        .Case<fir::AllocaOp, fir::AllocMemOp>([&](auto op) {
          // Unique memory allocation.
          type = SourceKind::Allocate;
          breakFromLoop = true;
        })
        .Case<fir::ConvertOp>([&](auto op) {
          // Skip ConvertOp's and track further through the operand.
          v = op->getOperand(0);
          defOp = v.getDefiningOp();
        })
        .Case<fir::BoxAddrOp>([&](auto op) {
          v = op->getOperand(0);
          defOp = v.getDefiningOp();
          if (mlir::isa<fir::BaseBoxType>(v.getType()))
            followBoxData = true;
        })
        .Case<fir::ArrayCoorOp, fir::CoordinateOp>([&](auto op) {
          if (isPointerReference(ty))
            attributes.set(Attribute::Pointer);
          v = op->getOperand(0);
          defOp = v.getDefiningOp();
          if (mlir::isa<fir::BaseBoxType>(v.getType()))
            followBoxData = true;
          approximateSource = true;
        })
        .Case<fir::EmboxOp, fir::ReboxOp>([&](auto op) {
          if (followBoxData) {
            v = op->getOperand(0);
            defOp = v.getDefiningOp();
          } else
            breakFromLoop = true;
        })
        .Case<fir::LoadOp>([&](auto op) {
          // If the load is from a leaf source, return the leaf. Do not track
          // through indirections otherwise.
          // TODO: Add support to fir.alloca and fir.allocmem
          auto def = getOriginalDef(op.getMemref());
          if (isDummyArgument(def) ||
              def.template getDefiningOp<fir::AddrOfOp>()) {
            v = def;
            defOp = v.getDefiningOp();
            return;
          }
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
          // No further tracking for addresses loaded from memory for now.
          type = SourceKind::Indirect;
          breakFromLoop = true;
        })
        .Case<fir::AddrOfOp>([&](auto op) {
          // Address of a global scope object.
          ty = v.getType();
          type = SourceKind::Global;

          auto globalOpName = mlir::OperationName(
              fir::GlobalOp::getOperationName(), defOp->getContext());
          if (fir::valueHasFirAttribute(
                  v, fir::GlobalOp::getTargetAttrName(globalOpName)))
            attributes.set(Attribute::Target);

          // TODO: Take followBoxData into account when setting the pointer
          // attribute
          if (isPointerReference(ty))
            attributes.set(Attribute::Pointer);
          global = llvm::cast<fir::AddrOfOp>(op).getSymbol();
          breakFromLoop = true;
        })
        .Case<hlfir::DeclareOp, fir::DeclareOp>([&](auto op) {
          bool isPrivateItem = false;
          if (omp::BlockArgOpenMPOpInterface argIface =
                  dyn_cast<omp::BlockArgOpenMPOpInterface>(op->getParentOp())) {
            Value ompValArg;
            llvm::TypeSwitch<Operation *>(op->getParentOp())
                .template Case<omp::TargetOp>([&](auto targetOp) {
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
                               omp::TaskloopOp, omp::TaskOp, omp::WsloopOp>(
                    [&](auto privateOp) {
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
            v = defOp->getResult(0);
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
        .Case<hlfir::DesignateOp>([&](auto op) {
          auto varIf = llvm::cast<fir::FortranVariableOpInterface>(defOp);
          attributes |= getAttrsFromVariable(varIf);
          // Track further through the memory indexed into
          // => if the source arrays/structures don't alias then nor do the
          //    results of hlfir.designate
          v = op.getMemref();
          defOp = v.getDefiningOp();
          // TODO: there will be some cases which provably don't alias if one
          // takes into account the component or indices, which are currently
          // ignored here - leading to false positives
          // because of this limitation, we need to make sure we never return
          // MustAlias after going through a designate operation
          approximateSource = true;
          if (mlir::isa<fir::BaseBoxType>(v.getType()))
            followBoxData = true;
        })
        .Default([&](auto op) {
          defOp = nullptr;
          breakFromLoop = true;
        });
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
    }
  }

  if (type == SourceKind::Global) {
    return {{global, instantiationPoint, followingData},
            type,
            ty,
            attributes,
            approximateSource,
            isCapturedInInternalProcedure};
  }
  return {{v, instantiationPoint, followingData},
          type,
          ty,
          attributes,
          approximateSource,
          isCapturedInInternalProcedure};
}

} // namespace fir
