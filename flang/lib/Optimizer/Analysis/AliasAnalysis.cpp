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
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AliasAnalysis: alias
//===----------------------------------------------------------------------===//

static bool isDummyArgument(mlir::Value v) {
  auto blockArg{v.dyn_cast<mlir::BlockArgument>()};
  if (!blockArg)
    return false;

  return blockArg.getOwner()->isEntryBlock();
}

namespace fir {

void AliasAnalysis::Source::print(llvm::raw_ostream &os) const {
  if (auto v = llvm::dyn_cast<mlir::Value>(u))
    os << v;
  else if (auto gbl = llvm::dyn_cast<mlir::SymbolRefAttr>(u))
    os << gbl;
  os << " SourceKind: " << EnumToString(kind);
  os << " Type: " << valueType << " ";
  attributes.Dump(os, EnumToString);
}

bool AliasAnalysis::Source::isPointerReference(mlir::Type ty) {
  auto eleTy = fir::dyn_cast_ptrEleTy(ty);
  if (!eleTy)
    return false;

  return fir::isPointerType(eleTy) || eleTy.isa<fir::PointerType>();
}

bool AliasAnalysis::Source::isTargetOrPointer() const {
  return attributes.test(Attribute::Pointer) ||
         attributes.test(Attribute::Target);
}

bool AliasAnalysis::Source::isRecordWithPointerComponent() const {
  auto eleTy = fir::dyn_cast_ptrEleTy(valueType);
  if (!eleTy)
    return false;
  // TO DO: Look for pointer components
  return eleTy.isa<fir::RecordType>();
}

AliasResult AliasAnalysis::alias(Value lhs, Value rhs) {
  auto lhsSrc = getSource(lhs);
  auto rhsSrc = getSource(rhs);

  // Indirect case currently not handled. Conservatively assume
  // it aliases with everything
  if (lhsSrc.kind == SourceKind::Indirect ||
      lhsSrc.kind == SourceKind::Unknown ||
      rhsSrc.kind == SourceKind::Indirect || rhsSrc.kind == SourceKind::Unknown)
    return AliasResult::MayAlias;

  if (lhsSrc.kind == rhsSrc.kind) {
    if (lhsSrc.u == rhsSrc.u)
      return AliasResult::MustAlias;

    // Allocate and global memory address cannot physically alias
    if (lhsSrc.kind == SourceKind::Allocate ||
        lhsSrc.kind == SourceKind::Global)
      return AliasResult::NoAlias;

    assert(lhsSrc.kind == SourceKind::Argument &&
           "unexpected memory source kind");

    // Dummy TARGET/POINTER arguments may alias.
    if (lhsSrc.isTargetOrPointer() && rhsSrc.isTargetOrPointer())
      return AliasResult::MayAlias;

    // Box for POINTER component inside an object of a derived type
    // may alias box of a POINTER object, as well as boxes for POINTER
    // components inside two objects of derived types may alias.
    if ((lhsSrc.isRecordWithPointerComponent() && rhsSrc.isTargetOrPointer()) ||
        (rhsSrc.isRecordWithPointerComponent() && lhsSrc.isTargetOrPointer()) ||
        (lhsSrc.isRecordWithPointerComponent() &&
         rhsSrc.isRecordWithPointerComponent()))
      return AliasResult::MayAlias;

    return AliasResult::NoAlias;
  }

  assert(lhsSrc.kind != rhsSrc.kind && "memory source kinds must be the same");

  Source *src1, *src2;
  if (lhsSrc.kind < rhsSrc.kind) {
    src1 = &lhsSrc;
    src2 = &rhsSrc;
  } else {
    src1 = &rhsSrc;
    src2 = &lhsSrc;
  }

  assert(src2->kind <= SourceKind::Argument && "unexpected memory source kind");
  if (src1->kind == SourceKind::Allocate)
    return AliasResult::NoAlias;

  assert(src1->kind == SourceKind::Global &&
         src2->kind == SourceKind::Argument &&
         "unexpected memory source kinds");

  // Dummy TARGET/POINTER argument may alias with a global TARGET/POINTER.
  if (src1->isTargetOrPointer() && src2->isTargetOrPointer())
    return AliasResult::MayAlias;

  // Box for POINTER component inside an object of a derived type
  // may alias box of a POINTER object, as well as boxes for POINTER
  // components inside two objects of derived types may alias.
  if ((src1->isRecordWithPointerComponent() && src2->isTargetOrPointer()) ||
      (src2->isRecordWithPointerComponent() && src1->isTargetOrPointer()) ||
      (src1->isRecordWithPointerComponent() &&
       src2->isRecordWithPointerComponent()))
    return AliasResult::MayAlias;

  return AliasResult::NoAlias;
}

//===----------------------------------------------------------------------===//
// AliasAnalysis: getModRef
//===----------------------------------------------------------------------===//

/// This is mostly inspired by MLIR::LocalAliasAnalysis with 2 notable
/// differences 1) Regions are not handled here but will be handled by a data
/// flow analysis to come 2) Allocate and Free effects are considered
/// modifying
ModRefResult AliasAnalysis::getModRef(Operation *op, Value location) {
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return ModRefResult::getModAndRef();

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

AliasAnalysis::Source AliasAnalysis::getSource(mlir::Value v) {
  auto *defOp = v.getDefiningOp();
  SourceKind type{SourceKind::Unknown};
  mlir::Type ty;
  bool breakFromLoop{false};
  mlir::SymbolRefAttr global;
  Source::Attributes attributes;
  while (defOp && !breakFromLoop) {
    ty = defOp->getResultTypes()[0];
    llvm::TypeSwitch<Operation *>(defOp)
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
        .Case<fir::LoadOp>([&](auto op) {
          // No further tracking for addresses loaded from memory (e.g. a box)
          // right now.
          type = SourceKind::Indirect;
          breakFromLoop = true;
        })
        .Case<fir::AddrOfOp>([&](auto op) {
          // Address of a global scope object.
          type = SourceKind::Global;
          ty = v.getType();
          if (fir::valueHasFirAttribute(v,
                                        fir::GlobalOp::getTargetAttrNameStr()))
            attributes.set(Attribute::Target);

          if (Source::isPointerReference(ty))
            attributes.set(Attribute::Pointer);
          global = llvm::cast<fir::AddrOfOp>(op).getSymbol();
          breakFromLoop = true;
        })
        .Default([&](auto op) {
          defOp = nullptr;
          breakFromLoop = true;
        });
  }
  if (!defOp && type == SourceKind::Unknown)
    // Check if the memory source is coming through a dummy argument.
    if (isDummyArgument(v)) {
      type = SourceKind::Argument;
      ty = v.getType();
      if (fir::valueHasFirAttribute(v, fir::getTargetAttrName()))
        attributes.set(Attribute::Target);

      if (Source::isPointerReference(ty))
        attributes.set(Attribute::Pointer);
    }

  if (type == SourceKind::Global)
    return {global, type, ty, attributes};

  return {v, type, ty, attributes};
}

} // namespace fir
