//===- CIRBasicAliasAnalysis.cpp - Basic CIR Alias Analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "cir-basic-alias-analysis"

using namespace llvm;
using namespace cir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static constexpr unsigned MaxLookupDepth = 6;

mlir::Value CIRBasicAliasAnalysis::getUnderlyingObject(mlir::Value val) {
  LDBG() << "Getting underlying object for: " << val;

  for (unsigned depth = 0; depth < MaxLookupDepth; ++depth) {
    mlir::Operation *defOp = val.getDefiningOp();
    if (!defOp) {
      LDBG() << "No defining operation, stopping";
      break; // Block argument (e.g. function parameter) — stop here.
    }

    // Bitcast and address-space casts don't change the underlying object.
    // array_to_ptrdecay produces an element pointer to the same storage as
    // the array pointer, so strip through it too.
    if (auto castOp = mlir::dyn_cast<cir::CastOp>(defOp)) {
      if (castOp.isAllocaPreservingCast() ||
          castOp.getKind() == cir::CastKind::array_to_ptrdecay) {
        LDBG() << "Walking past cast operation";
        val = castOp.getSrc();
        continue;
      }
      LDBG() << "Opaque cast operation, stopping";
      break;
    }

    // Pointer stride: only strip through when we can prove the access stays
    // within the bounds of the underlying allocation.
    if (auto strideOp = mlir::dyn_cast<cir::PtrStrideOp>(defOp)) {
      auto constOp = strideOp.getStride().getDefiningOp<cir::ConstantOp>();
      if (constOp) {
        if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(constOp.getValue())) {
          APInt stride = intAttr.getValue();

          // Zero stride is trivially in-bounds.
          if (stride.isZero()) {
            LDBG() << "Walking past zero-strided PtrStrideOp";
            val = strideOp.getBase();
            continue;
          }
        }
      }
      // Dynamic stride or unverifiable bounds — stop here conservatively.
      LDBG() << "Non-zero or dynamic PtrStrideOp, stopping";
      break;
    }

    // Handle special cases for zero-offset sub-object accesses.
    if (auto op = mlir::dyn_cast<cir::GetMemberOp>(defOp)) {
      if (op.getIndex() == 0) {
        LDBG() << "GetMemberOp[0], following to underlying object";
        val = op.getAddr();
        continue;
      } else {
        LDBG() << "GetMemberOp, non-zero index, stopping";
        break;
      }
    }
    if (auto op = mlir::dyn_cast<cir::GetElementOp>(defOp)) {
      cir::IntAttr index;
      if (auto constOp = op.getIndex().getDefiningOp<cir::ConstantOp>())
        index = mlir::dyn_cast<cir::IntAttr>(constOp.getValue());
      if (index && index.getValue().isZero()) {
        LDBG() << "GetElementOp[0], following to underlying object";
        val = op.getBase();
        continue;
      }
      LDBG() << "GetElementOp, non-zero or dynamic index, stopping";
      break;
    }
    if (auto op = mlir::dyn_cast<cir::BaseClassAddrOp>(defOp)) {
      // A zero byte offset means the base subobject starts at the same address
      // as the derived object.
      if (op.getOffset().isZero()) {
        LDBG() << "BaseClassAddrOp[0], following to underlying object";
        val = op.getDerivedAddr();
        continue;
      }
      LDBG() << "BaseClassAddrOp, non-zero offset, stopping";
      break;
    }
    if (auto op = mlir::dyn_cast<cir::DerivedClassAddrOp>(defOp)) {
      // The offset is stored unsigned but applied as a negative adjustment. A
      // zero offset means the derived object starts at the same address as the
      // base subobject.
      if (op.getOffset().isZero()) {
        LDBG() << "DerivedClassAddrOp[0], following to underlying object";
        val = op.getBaseAddr();
        continue;
      }
      LDBG() << "DerivedClassAddrOp, non-zero offset, stopping";
      break;
    }
    if (auto op = mlir::dyn_cast<cir::ComplexRealPtrOp>(defOp)) {
      LDBG() << "Getting input pointer for ComplexRealPtrOp";
      val = op.getOperand();
      continue;
    }
    if (auto op = mlir::dyn_cast<cir::ComplexImagPtrOp>(defOp)) {
      LDBG() << "ComplexImagPtrOp, stopping";
      break;
    }

    LDBG() << "Unhandled operation, stopping";
    break; // Unknown op — stop here conservatively.
  }
  return val;
}

CIRBasicAliasAnalysis::ObjectRelation
CIRBasicAliasAnalysis::classifyObjects(mlir::Value lhs, mlir::Value rhs) {
  LDBG() << "Checking if " << lhs << " and " << rhs << " are distinct objects";

  // Two values are distinct allocations if they originate from different
  // cir.alloca operations (or other allocation ops) in the same function.
  // TODO: Extend to cover global addresses, function arguments with noalias,
  // and heap allocations.
  mlir::Value lhsObj = getUnderlyingObject(lhs);
  mlir::Value rhsObj = getUnderlyingObject(rhs);

  if (lhsObj == rhsObj) {
    LDBG() << "Identical values, not distinct";
    return ObjectRelation::Identical;
  }

  // Different cir.alloca ops in the same function cannot alias.
  if (mlir::isa_and_nonnull<cir::AllocaOp>(lhsObj.getDefiningOp()) &&
      mlir::isa_and_nonnull<cir::AllocaOp>(rhsObj.getDefiningOp())) {
    LDBG() << "Different cir.alloca ops in the same function, distinct";
    return ObjectRelation::Distinct;
  }

  LDBG() << "Conservative fallback, not distinct";
  return ObjectRelation::Unknown;
}

//===----------------------------------------------------------------------===//
// CIRBasicAliasAnalysis
//===----------------------------------------------------------------------===//

mlir::AliasResult CIRBasicAliasAnalysis::alias(mlir::Value lhs,
                                               mlir::Value rhs) {
  LDBG() << "Checking alias between: " << lhs << " and " << rhs;

  if (lhs == rhs) {
    LDBG() << "Trivial alias between identical values";
    return mlir::AliasResult::MustAlias;
  }

  ObjectRelation relation = classifyObjects(lhs, rhs);
  switch (relation) {
  case ObjectRelation::Distinct:
    LDBG() << "No alias between distinct objects";
    return mlir::AliasResult::NoAlias;
  case ObjectRelation::Identical:
    LDBG() << "Must alias between identical objects";
    return mlir::AliasResult::MustAlias;
  case ObjectRelation::Unknown:
    // Conservative fallback — the aggregate will try other implementations.
    LDBG() << "Conservative fallback, may alias";
    return mlir::AliasResult::MayAlias;
  }
  llvm_unreachable("Unhandled ObjectRelation");
}

mlir::ModRefResult CIRBasicAliasAnalysis::getModRef(mlir::Operation *op,
                                                    mlir::Value location) {
  LDBG() << "getModRef: "
         << mlir::OpWithFlags(op, mlir::OpPrintingFlags().skipRegions())
         << " on location " << location;

  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effects) {
    LDBG() << "No memory effect interface, returning ModAndRef";
    return mlir::ModRefResult::getModAndRef();
  }

  SmallVector<mlir::MemoryEffects::EffectInstance> effectList;
  effects.getEffects(effectList);

  auto classifyEffect = [location, this](
                            const mlir::MemoryEffects::EffectInstance &effect) {
    if (mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect())) {
      LDBG() << "Skipping allocate effect";
      return mlir::ModRefResult::getNoModRef();
    }

    mlir::AliasResult aliasResult = mlir::AliasResult::MayAlias;
    if (mlir::Value affectedLocation = effect.getValue()) {
      LDBG() << "    Checking alias between affected location "
             << affectedLocation << " and query location " << location;
      aliasResult = alias(affectedLocation, location);
      LDBG() << "    Alias result: "
             << (aliasResult.isMust() ? "MustAlias"
                 : aliasResult.isNo() ? "NoAlias"
                                      : "MayAlias");
    } else {
      // An effect on a non-addressable resource cannot affect a
      // pointer-based location.
      if (!effect.getResource()->isAddressable()) {
        LDBG() << "    Effect on non-addressable resource '"
               << effect.getResource()->getName() << "', skipping (NoAlias)";
        aliasResult = mlir::AliasResult::NoAlias;
      } else {
        LDBG() << "    No effect value, assuming MayAlias";
      }
    }

    // If the affected location doesn't alias with the query location,
    // ignore this effect.
    if (aliasResult.isNo()) {
      LDBG() << "No alias with affected location";
      return mlir::ModRefResult::getNoModRef();
    }

    // TODO: Consider whether Free should be NoModRef.
    if (mlir::isa<mlir::MemoryEffects::Free>(effect.getEffect())) {
      LDBG() << "Skipping free effect";
      return mlir::ModRefResult::getModAndRef();
    }

    if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      LDBG() << "Write effect, adding Mod";
      return mlir::ModRefResult::getMod();
    }

    if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
      LDBG() << "Read effect, adding Ref";
      return mlir::ModRefResult::getRef();
    }

    LDBG() << "Unexpected memory effect: " << effect.getEffect();
    return mlir::ModRefResult::getNoModRef();
  };

  return llvm::accumulate(llvm::map_range(effectList, classifyEffect),
                          mlir::ModRefResult::getNoModRef(),
                          [](mlir::ModRefResult lhs, mlir::ModRefResult rhs) {
                            return lhs.merge(rhs);
                          });
}
