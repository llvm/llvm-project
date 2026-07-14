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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <utility>

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
            /*isCapturedInInternalProcedure=*/false,
            /*scopedOrigins=*/{}};

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

/// Predecessor SSA values that may define a result of \p branch when control
/// continues in the parent region (same mapping as
/// `LocalAliasAnalysis::collectUnderlyingAddressValues2` for
/// `RegionSuccessor(branch.getOperation())`).
static void getRegionBranchPredecessorValuesForParentResult(
    mlir::RegionBranchOpInterface branch, mlir::OpResult result,
    llvm::SmallVectorImpl<mlir::Value> &out) {
  mlir::RegionSuccessor parentSucc(branch.getOperation());
  mlir::Value inputValue = result;
  unsigned inputIndex = result.getResultNumber();
  mlir::ValueRange inputs = branch.getSuccessorInputs(parentSucc);
  if (inputs.empty()) {
    out.push_back(inputValue);
    return;
  }
  unsigned firstInputIndex, lastInputIndex;
  if (mlir::isa<mlir::BlockArgument>(inputs[0])) {
    firstInputIndex = mlir::cast<mlir::BlockArgument>(inputs[0]).getArgNumber();
    lastInputIndex =
        mlir::cast<mlir::BlockArgument>(inputs.back()).getArgNumber();
  } else {
    firstInputIndex = mlir::cast<mlir::OpResult>(inputs[0]).getResultNumber();
    lastInputIndex =
        mlir::cast<mlir::OpResult>(inputs.back()).getResultNumber();
  }
  if (firstInputIndex > inputIndex || lastInputIndex < inputIndex) {
    out.push_back(inputValue);
    return;
  }
  branch.getPredecessorValues(parentSucc, inputIndex - firstInputIndex, out);
}

/// True when \p src's tracked origin value is an SSA result of an operation
/// nested under \p branch's regions.
static bool originIsInsideRegionBranch(mlir::RegionBranchOpInterface branch,
                                       const fir::AliasAnalysis::Source &src) {
  const fir::AliasAnalysis::Source::SourceOrigin &origin = src.origin;
  if (llvm::isa<mlir::SymbolRefAttr>(origin.u))
    return false;
  mlir::Value originVal = llvm::cast<mlir::Value>(origin.u);
  if (mlir::isa<mlir::BlockArgument>(originVal))
    return false;
  mlir::Operation *defOp = originVal.getDefiningOp();
  if (!defOp)
    return false;
  return branch.getOperation()->isProperAncestor(defOp);
}

/// Conservative join of memory sources from region-branch predecessors.
static fir::AliasAnalysis::Source mergeRegionBranchPredecessorSources(
    llvm::ArrayRef<fir::AliasAnalysis::Source> sources,
    mlir::Value fallbackValue, mlir::Type fallbackType, bool followingData) {
  assert(!sources.empty() && "expected at least one predecessor source");

  // For kind, origin, attributes, isApproximate/accessPath, valueType, we
  // capture if all of the sources have exactly the same value.
  bool allKindsSame =
      llvm::all_of(sources, [&](const fir::AliasAnalysis::Source &s) {
        return s.kind == sources[0].kind;
      });
  bool allOriginsSame =
      llvm::all_of(sources, [&](const fir::AliasAnalysis::Source &s) {
        return s.origin == sources[0].origin;
      });
  bool allAttrsSame =
      llvm::all_of(sources, [&](const fir::AliasAnalysis::Source &s) {
        return s.attributes == sources[0].attributes;
      });
  bool allPathsSame =
      llvm::all_of(sources, [&](const fir::AliasAnalysis::Source &s) {
        return s.accessPath == sources[0].accessPath;
      });
  bool allTypesSame =
      llvm::all_of(sources, [&](const fir::AliasAnalysis::Source &s) {
        return s.valueType == sources[0].valueType;
      });

  // For approximateSource and isCapturedInInternalProcedure, we mark them
  // as true if any of the sources are true.
  bool mergedApprox =
      llvm::any_of(sources, [](const fir::AliasAnalysis::Source &s) {
        return s.approximateSource;
      });
  bool mergedCaptured =
      llvm::any_of(sources, [](const fir::AliasAnalysis::Source &s) {
        return s.isCapturedInInternalProcedure;
      });

  fir::AliasAnalysis::SourceKind mergedKind;
  fir::AliasAnalysis::Source::Attributes mergedAttrs;
  if (!allKindsSame) {
    mergedKind = fir::AliasAnalysis::SourceKind::Unknown;
    mergedAttrs = {};
  } else if (!allAttrsSame) {
    mergedKind = fir::AliasAnalysis::SourceKind::Unknown;
    mergedAttrs = {};
  } else if (!allOriginsSame) {
    // Same kind and attributes on every path, but different concrete origins.
    // Since origins are different, for most cases fall back to Indirect here.
    // However, for Allocate, we want to keep the information about this being
    // an Allocate as long as all are defined inside the region's branches
    // because then they are all unique and thus cannot alias anything outside
    // the region (this is key here - because this only holds when comparing
    // region's result only with outside values not the origins themselves).
    // TODO: An origin list would be better to preserve this information
    // more accurately instead of a single origin.
    auto branchOp = mlir::dyn_cast<mlir::RegionBranchOpInterface>(
        mlir::cast<mlir::OpResult>(fallbackValue).getOwner());
    assert(branchOp && "merge region-branch sources expects branch op result");
    bool hasOriginOutsideBranch =
        llvm::any_of(sources, [&](const fir::AliasAnalysis::Source &s) {
          return !originIsInsideRegionBranch(branchOp, s);
        });
    bool keepAllocate =
        sources[0].kind == fir::AliasAnalysis::SourceKind::Allocate &&
        !hasOriginOutsideBranch;
    mergedKind = keepAllocate ? fir::AliasAnalysis::SourceKind::Allocate
                              : fir::AliasAnalysis::SourceKind::Indirect;
    mergedAttrs = sources[0].attributes;
  } else {
    mergedKind = sources[0].kind;
    mergedAttrs = sources[0].attributes;
  }

  fir::AliasAnalysis::Source::SourceOrigin mergedOrigin;
  if (allOriginsSame) {
    mergedOrigin = sources[0].origin;
  } else {
    // Set the origin as the fallbackValue provided - which should be the
    // region-branch result.
    mergedOrigin = {fallbackValue, nullptr, followingData};
  }

  fir::AliasAnalysis::Source::AccessPath mergedPath;
  if (allPathsSame) {
    mergedPath = sources[0].accessPath;
    mergedPath.isApproximate |= mergedApprox;
  } else {
    mergedPath = {};
    mergedPath.isApproximate = true;
  }

  mlir::Type mergedTy = allTypesSame ? sources[0].valueType : fallbackType;

  // Intersect scopedOrigins across predecessors by declValue. The
  // declare's governing scope is a deterministic function of the
  // declare op, so declValue alone identifies the snapshot (the same
  // declValue can never carry two different scopes). For an entry that
  // matches in every predecessor, take the bitwise union (|=) of its
  // 'attributes' and 'approximateSource' bits and keep the first
  // predecessor's path steps (the steps must be equal to match, so
  // there is nothing to merge there). Drop the entry on a path-step or
  // isData mismatch, because different control-flow shapes from the
  // same declare to the merge point cannot be summarised safely.
  llvm::SmallVector<fir::AliasAnalysis::Source::ScopedOrigin, 4>
      mergedScopedOrigins;
  if (!sources.empty()) {
    // Seed with the first predecessor's snapshots, keyed by declValue
    // for intersection lookups.
    llvm::DenseMap<void *, unsigned> indexInMerged;
    mergedScopedOrigins.assign(sources[0].scopedOrigins.begin(),
                               sources[0].scopedOrigins.end());
    for (unsigned i = 0; i < mergedScopedOrigins.size(); ++i) {
      const auto &scopedOrigin = mergedScopedOrigins[i];
      indexInMerged[scopedOrigin.declValue.getAsOpaquePointer()] = i;
    }
    llvm::SmallVector<bool, 4> seenThisPred;
    for (unsigned predIdx = 1; predIdx < sources.size(); ++predIdx) {
      seenThisPred.assign(mergedScopedOrigins.size(), false);
      for (const auto &scopedOrigin : sources[predIdx].scopedOrigins) {
        auto it =
            indexInMerged.find(scopedOrigin.declValue.getAsOpaquePointer());
        if (it == indexInMerged.end())
          continue;
        unsigned idx = it->second;
        auto &mergedScopedOrigin = mergedScopedOrigins[idx];
        if (mergedScopedOrigin.isData != scopedOrigin.isData ||
            mergedScopedOrigin.accessPath.steps !=
                scopedOrigin.accessPath.steps) {
          // Mark as not-seen so the entry is dropped after this pred.
          continue;
        }
        mergedScopedOrigin.attributes |= scopedOrigin.attributes;
        mergedScopedOrigin.approximateSource |= scopedOrigin.approximateSource;
        mergedScopedOrigin.accessPath.isApproximate |=
            scopedOrigin.accessPath.isApproximate;
        seenThisPred[idx] = true;
      }
      // Drop entries this predecessor did not match. Iterate in reverse
      // to keep earlier indices valid while erasing.
      for (int idx = static_cast<int>(mergedScopedOrigins.size()) - 1; idx >= 0;
           --idx) {
        if (!seenThisPred[idx]) {
          const auto &scopedOrigin = mergedScopedOrigins[idx];
          indexInMerged.erase(scopedOrigin.declValue.getAsOpaquePointer());
          mergedScopedOrigins.erase(mergedScopedOrigins.begin() + idx);
        }
      }
      // Recompute indices after erase shifts.
      indexInMerged.clear();
      for (unsigned i = 0; i < mergedScopedOrigins.size(); ++i) {
        const auto &scopedOrigin = mergedScopedOrigins[i];
        indexInMerged[scopedOrigin.declValue.getAsOpaquePointer()] = i;
      }
    }
  }

  return {
      mergedOrigin, mergedKind, mergedTy,       mergedAttrs,
      mergedApprox, mergedPath, mergedCaptured, std::move(mergedScopedOrigins)};
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
  // through other dialects.
  //
  // Scope-aware refinement is only meaningful after inlining, when the
  // function contains more than one fir.dummy_scope op. Skip
  // collectScopedOrigins and the scope-pair loop for non-inlined functions
  // to avoid the per-query getDeclarationScope/DominanceInfo overhead.
  bool multiScopes = functionHasMultipleScopes(lhs);
  auto lhsSrc =
      getSource(lhs, /*getLastInstantiationPoint=*/false, multiScopes);
  auto rhsSrc =
      getSource(rhs, /*getLastInstantiationPoint=*/false, multiScopes);
  AliasResult result = alias(lhsSrc, rhsSrc, lhs, rhs);

  // Scope-aware refinement after inlining: if both walks crossed declares
  // in the SAME Fortran procedure scope at DISTINCT declare values, the
  // two queries may still be disambiguated by rebuilding intermediate
  // Sources rooted at each shared-scope declare pair (Fortran 2018
  // 15.5.2.13: distinct dummy arguments / locals of the same procedure
  // frame do not alias unless TARGET/POINTER/Cray attributes permit it).
  // The per-pair check delegates back to the 4-arg alias() with paths and
  // attributes snapshotted at the declares, so TARGET-attributed declares
  // and pointer-dereferenced paths remain correctly reported as MayAlias.
  // Short-circuit on NoAlias since any pair that disambiguates is
  // decisive.
  if (!multiScopes || result == AliasResult::NoAlias ||
      result == AliasResult::MustAlias)
    return result;
  for (const auto &lhsScopedOrigin : lhsSrc.scopedOrigins) {
    if (!lhsScopedOrigin.scope)
      continue;
    for (const auto &rhsScopedOrigin : rhsSrc.scopedOrigins) {
      if (lhsScopedOrigin.scope != rhsScopedOrigin.scope ||
          lhsScopedOrigin.declValue == rhsScopedOrigin.declValue)
        continue;
      Source lhsInner = buildSourceAtDeclare(lhsScopedOrigin);
      Source rhsInner = buildSourceAtDeclare(rhsScopedOrigin);
      // Use the OUTER lhs/rhs values, not the declare values: the
      // rebuilt Sources describe accessing the outer query's leaf
      // through the captured declare. Passing the declares here would
      // make type-based checks (e.g. descriptor-vs-data via
      // noAliasBasedOnType) compare the declare's box-descriptor type
      // to the outer leaf type and yield bogus NoAlias for pointer
      // dereferences.
      AliasResult refined = alias(lhsInner, rhsInner, lhs, rhs);
      if (refined == AliasResult::NoAlias) {
        LLVM_DEBUG(llvm::dbgs() << "  no alias via scoped-origin refinement\n");
        return AliasResult::NoAlias;
      }
    }
  }
  return result;
}

AliasResult AliasAnalysis::alias(Source lhsSrc, Source rhsSrc, mlir::Value lhs,
                                 mlir::Value rhs) {
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
  // Reuse this AliasAnalysis instance instead of constructing a fresh one per
  // query. getCallModRef is only reached from getModRef, and getSource/alias
  // never call getModRef, so there is no recursion. varSrc is used only to
  // classify var (kind/attributes); getCallModRef never inspects its
  // scopedOrigins, so skip their (potentially expensive) collection.
  fir::AliasAnalysis::Source varSrc =
      getSource(var, /*getLastInstantiationPoint=*/true,
                /*collectScopedOrigins=*/false);
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
    if (fir::conformsWithPassByRef(arg.getType()) && !alias(arg, var).isNo()) {
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

/// Walk through any pass-through block-arg<->operand links the analysis
/// understands, replacing \p v with the corresponding operand at each step,
/// and return the resulting value. A "pass-through block argument" is one
/// that does not introduce a new value relative to its corresponding operand
/// from the standpoint of memory addressing, so walking past it is safe for
/// alias analysis.
///
/// Currently the only handled pass-through link is the
/// operand<->block-argument mapping of an acc.compute_region.
static mlir::Value walkBlockArgPassThroughs(mlir::Value v) {
  while (v) {
    mlir::Value operand = mlir::acc::getACCOperandForBlockArg(v);
    if (!operand)
      break;
    v = operand;
  }
  return v;
}

void AliasAnalysis::enableSourceCache() { sourceCacheEnabled = true; }

void AliasAnalysis::disableSourceCache() {
  sourceCacheEnabled = false;
  clearSourceCache();
}

AliasAnalysis::Source AliasAnalysis::getSource(mlir::Value v,
                                               bool getLastInstantiationPoint,
                                               bool collectScopedOrigins) {
  if (!sourceCacheEnabled)
    return getSourceImpl(v, getLastInstantiationPoint, collectScopedOrigins);

  // Key on the queried value and the two boolean flags. Recursive sub-queries
  // go through this same wrapper, so the whole walk is memoized.
  std::pair<mlir::Value, unsigned> key{v,
                                       (getLastInstantiationPoint ? 1u : 0u) |
                                           (collectScopedOrigins ? 2u : 0u)};
  auto it = getSourceCache.find(key);
  if (it != getSourceCache.end()) {
    ++sourceCacheHits;
    return it->second;
  }

  ++sourceCacheMisses;
  Source source =
      getSourceImpl(v, getLastInstantiationPoint, collectScopedOrigins);
  getSourceCache.try_emplace(key, source);
  return source;
}

AliasAnalysis::Source
AliasAnalysis::getSourceImpl(mlir::Value v, bool getLastInstantiationPoint,
                             bool collectScopedOrigins) {
  // If v is a pass-through block argument (see walkBlockArgPassThroughs),
  // continue from the underlying operand so the tracking loop below has a
  // defining op to chew on. Without this, a recursive query like the one in
  // the fir.load (box) branch below would immediately return
  // SourceKind::Unknown (no defining op and not a function dummy argument),
  // which then forces SourceKind::Indirect for box loads from such block args
  // and pessimizes alias analysis.
  v = walkBlockArgPassThroughs(v);
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

  // Per-declare snapshots collected as the walk crosses [hl]fir.declare ops.
  // Ordered from leaf-closest (front) to root-closest (back). Forwarded
  // through region-branch merges and the box-load branch, then threaded into
  // the final Source. Gated on collectScopedOrigins (suppressed when
  // buildSourceAtDeclare reuses getSource purely for declare classification).
  llvm::SmallVector<Source::ScopedOrigin, 4> scopedOrigins;
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
    std::optional<AliasAnalysis::Source> regionBranchReturn;
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

            // Keep the inner walk's getLastInstantiationPoint=false so it
            // continues past dummy-scope declares to the underlying
            // BlockArgument. The outer classification below relies on
            // boxSrc.origin.u being the BlockArg (so isDummyArgument()
            // succeeds and the outer SourceKind becomes Argument).
            // Passing true here would stop the inner walk at the declare
            // and force SourceKind::Indirect, which spuriously coarsens
            // getCallModRef (e.g. for box_addr of allocatable dummies).
            auto boxSrc = getSource(op.getMemref(),
                                    /*getLastInstantiationPoint=*/false,
                                    collectScopedOrigins);
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

            // Rebase each forwarded ScopedOrigin from the inner walk's
            // coordinate system (rooted at the inner declare, leaf=memref)
            // to the outer (leaf=original query) by splicing the deref
            // step and the outer pathSteps onto the snapshot's path.
            if (collectScopedOrigins) {
              for (auto scopedOrigin : boxSrc.scopedOrigins) {
                scopedOrigin.accessPath.steps.push_back(derefStep);
                for (int i = pathSteps.size() - 1; i >= 0; --i)
                  scopedOrigin.accessPath.steps.push_back(pathSteps[i]);
                scopedOrigin.accessPath.isApproximate |= approximateSource;
                if (isPointerBox)
                  scopedOrigin.attributes.set(Attribute::Pointer);
                scopedOrigin.approximateSource |= approximateSource;
                // The inner walk computed isData against the box memref
                // (typically false, since the walk started at
                // !fir.ref<!fir.box<...>>). After splicing the deref step, the
                // snapshot's path now describes reaching the outer query's leaf
                // via the box's pointer, so it follows data iff the outer walk
                // does.
                scopedOrigin.isData = followingData;
                scopedOrigins.push_back(std::move(scopedOrigin));
              }
            }

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
                } else if (boxSrc.kind == SourceKind::HostAssoc) {
                  // Box loaded from a host-associated descriptor: classify
                  // the dereferenced target as HostAssoc (not Indirect) so
                  // alias() can apply the host-assoc/pointer rules instead
                  // of coarsening to MayAlias. The access path (PointerDeref/
                  // AllocDeref step) and Pointer attribute were already set
                  // above, so the resulting Source matches the one that
                  // buildSourceAtDeclare() rebuilds during scope-aware
                  // refinement.
                  type = SourceKind::HostAssoc;
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

          // Snapshot a ScopedOrigin at this declare. The snapshot
          // captures the path/attributes from the leaf to this declare
          // and is used by alias() for scope-aware refinement.
          if (collectScopedOrigins) {
            Source::ScopedOrigin scopedOrigin;
            scopedOrigin.scope = getDeclarationScope(op);
            scopedOrigin.declValue = opResult;
            scopedOrigin.accessPath.steps.assign(pathSteps.rbegin(),
                                                 pathSteps.rend());
            scopedOrigin.accessPath.isApproximate = approximateSource;
            scopedOrigin.attributes = attributes;
            scopedOrigin.approximateSource = approximateSource;
            scopedOrigin.isData = followingData;
            scopedOrigins.push_back(std::move(scopedOrigin));
          }

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
                return getSource(x, getLastInstantiationPoint,
                                 collectScopedOrigins);
              },
              followingData, attributes);
          breakFromLoop = true;
        })
        .Case([&](mlir::RegionBranchOpInterface branch) {
          llvm::SmallVector<mlir::Value, 4> predecessors;
          getRegionBranchPredecessorValuesForParentResult(branch, opResult,
                                                          predecessors);
          if (predecessors.empty() ||
              llvm::all_of(predecessors,
                           [&](mlir::Value pred) { return pred == v; })) {
            regionBranchReturn = {{{v, instantiationPoint, followingData},
                                   SourceKind::Unknown,
                                   ty,
                                   attributes,
                                   /*approximateSource=*/true,
                                   /*accessPath=*/{},
                                   isCapturedInInternalProcedure,
                                   /*scopedOrigins=*/{}}};
            breakFromLoop = true;
            return;
          }
          llvm::SmallVector<AliasAnalysis::Source, 4> predSources;
          predSources.reserve(predecessors.size());
          for (mlir::Value pred : predecessors)
            predSources.push_back(getSource(pred, getLastInstantiationPoint,
                                            collectScopedOrigins));
          regionBranchReturn = mergeRegionBranchPredecessorSources(
              predSources, v, ty, followingData);
          regionBranchReturn->attributes |= attributes;
          regionBranchReturn->approximateSource |= approximateSource;
          regionBranchReturn->isCapturedInInternalProcedure |=
              isCapturedInInternalProcedure;
          // Prepend the outer (leaf-closer) scopedOrigins -- declares
          // already crossed between leaf and this region-branch op --
          // to the merged predecessors' snapshots. The inner snapshots'
          // paths are relative to the region-branch result (matching
          // the existing approximation for the top-level accessPath
          // composed across region-branch merges).
          if (collectScopedOrigins && !scopedOrigins.empty()) {
            llvm::SmallVector<Source::ScopedOrigin, 4> combined;
            combined.reserve(scopedOrigins.size() +
                             regionBranchReturn->scopedOrigins.size());
            combined.append(scopedOrigins.begin(), scopedOrigins.end());
            combined.append(regionBranchReturn->scopedOrigins.begin(),
                            regionBranchReturn->scopedOrigins.end());
            regionBranchReturn->scopedOrigins = std::move(combined);
          }
          breakFromLoop = true;
        })
        .Default([&](auto op) {
          defOp = nullptr;
          breakFromLoop = true;
        });
    if (regionBranchReturn)
      return *regionBranchReturn;
    if (accSourceReturn)
      return *accSourceReturn;

    if (!breakFromLoop && v) {
      // If we have reached a pass-through block argument, walk past it so
      // the next loop iteration sees the underlying defining op.
      mlir::Value newV = walkBlockArgPassThroughs(v);
      if (newV != v) {
        v = newV;
        defOp = v.getDefiningOp();
      }
    }
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
            isCapturedInInternalProcedure,
            std::move(scopedOrigins)};
  }
  return {{v, instantiationPoint, followingData},
          type,
          ty,
          attributes,
          approximateSource,
          accessPath,
          isCapturedInInternalProcedure,
          std::move(scopedOrigins)};
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

mlir::Value
fir::AliasAnalysis::getDeclarationScope(mlir::Operation *declareOp) {
  assert(declareOp && "expected a non-null declare op");
  // Prefer the declare's explicit dummy_scope operand when present.
  if (auto hlfirDeclareOp = mlir::dyn_cast<hlfir::DeclareOp>(declareOp))
    if (mlir::Value dummyScope = hlfirDeclareOp.getDummyScope())
      return dummyScope;
  if (auto firDeclareOp = mlir::dyn_cast<fir::DeclareOp>(declareOp))
    if (mlir::Value dummyScope = firDeclareOp.getDummyScope())
      return dummyScope;

  // Otherwise look up the dominating fir.dummy_scope in the parent
  // function. Mirrors PassState::getDeclarationScope in AddAliasTags.cpp.
  auto func = declareOp->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return {};

  mlir::Operation *funcOp = func.getOperation();
  auto domIt = domInfoCache.find(funcOp);
  if (domIt == domInfoCache.end()) {
    auto inserted = domInfoCache.try_emplace(
        funcOp, std::make_unique<mlir::DominanceInfo>(funcOp));
    domIt = inserted.first;
  }
  mlir::DominanceInfo &domInfo = *domIt->second;

  auto scopeIt = sortedScopeCache.find(funcOp);
  if (scopeIt == sortedScopeCache.end()) {
    llvm::SmallVector<mlir::Operation *, 16> scopeOps;
    func.walk(
        [&](fir::DummyScopeOp op) { scopeOps.push_back(op.getOperation()); });
    llvm::stable_sort(scopeOps, [&](mlir::Operation *a, mlir::Operation *b) {
      return domInfo.properlyDominates(a, b);
    });
    scopeIt = sortedScopeCache.insert({funcOp, std::move(scopeOps)}).first;
  }

  const auto &scopeOps = scopeIt->second;
  for (auto it = scopeOps.rbegin(), ie = scopeOps.rend(); it != ie; ++it) {
    if (domInfo.dominates(*it, declareOp))
      return mlir::cast<fir::DummyScopeOp>(*it).getResult();
  }
  return {};
}

fir::AliasAnalysis::Source fir::AliasAnalysis::buildSourceAtDeclare(
    const fir::AliasAnalysis::Source::ScopedOrigin &scopedOrigin) {
  // Reuse getSource for classification (handles dummy_scope, alloca/
  // allocmem, address_of, etc. exactly as the main walk does). Disable
  // ScopedOrigin collection so we do not allocate snapshots that would
  // be immediately discarded.
  //
  // Pass getLastInstantiationPoint=true so the walk STOPS at the captured
  // declare and classifies the Source in that declare's own scope: a
  // dummy-scope declare becomes SourceKind::Argument, while a local/global
  // declare still continues to its alloca/address_of (Allocate/Global).
  // This is essential after inlining: with getLastInstantiationPoint=false
  // the walk would continue past the dummy declare through cross-scope
  // chains (e.g. fir.embox/fir.box_addr/scf.if introduced by contiguity
  // copy-in or OPTIONAL select in the caller frame), whose region-branch
  // merge collapses the kind to SourceKind::Unknown and makes the 4-arg
  // alias() report MayAlias -- defeating the whole point of the refinement.
  Source source = getSource(scopedOrigin.declValue,
                            /*getLastInstantiationPoint=*/true,
                            /*collectScopedOrigins=*/false);
  // Rebase path/attributes to the snapshot taken when the original
  // walk crossed this declare, so the returned Source represents
  // "declare-as-root, original-query-as-leaf".
  source.accessPath = scopedOrigin.accessPath;
  source.attributes = scopedOrigin.attributes;
  source.approximateSource = scopedOrigin.approximateSource;
  source.origin.isData = scopedOrigin.isData;
  return source;
}

bool fir::AliasAnalysis::functionHasMultipleScopes(mlir::Value v) {
  mlir::func::FuncOp funcOp;
  if (mlir::Operation *defOp = v.getDefiningOp())
    funcOp = defOp->getParentOfType<mlir::func::FuncOp>();
  else if (auto bArg = mlir::dyn_cast<mlir::BlockArgument>(v))
    if (mlir::Region *region = bArg.getOwner()->getParent())
      funcOp = region->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp)
    return true; // conservative
  mlir::Operation *funcOpPtr = funcOp.getOperation();
  auto it = multiScopeCache.find(funcOpPtr);
  if (it != multiScopeCache.end())
    return it->second;
  // Walk counting DummyScopeOps, stop early at 2.
  unsigned count = 0;
  funcOp.walk([&](fir::DummyScopeOp) -> mlir::WalkResult {
    return ++count >= 2 ? mlir::WalkResult::interrupt()
                        : mlir::WalkResult::advance();
  });
  // Cache both true and false so subsequent queries are O(1).
  return multiScopeCache.try_emplace(funcOpPtr, count >= 2).first->second;
}

} // namespace fir
