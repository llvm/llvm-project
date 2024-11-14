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
#include "mlir/Analysis/AliasAnalysis.h"
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

bool AliasAnalysis::Source::isData() const { return origin.isData; }
bool AliasAnalysis::Source::isBoxData() const {
  return mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(valueType)) &&
         origin.isData;
}

AliasResult AliasAnalysis::alias(mlir::Value lhs, mlir::Value rhs) {
  // TODO: alias() has to be aware of the function scopes.
  // After MLIR inlining, the current implementation may
  // not recognize non-aliasing entities.
  auto lhsSrc = getSource(lhs);
  auto rhsSrc = getSource(rhs);
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
    if (lhsSrc.origin == rhsSrc.origin) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  aliasing because same source kind and origin\n");
      if (approximateSource)
        return AliasResult::MayAlias;
      return AliasResult::MustAlias;
    }

    // Two host associated accesses may overlap due to an equivalence.
    if (lhsSrc.kind == SourceKind::HostAssoc) {
      LLVM_DEBUG(llvm::dbgs() << "  aliasing because of host association\n");
      return AliasResult::MayAlias;
    }
  }

  Source *src1, *src2;
  if (lhsSrc.kind < rhsSrc.kind) {
    src1 = &lhsSrc;
    src2 = &rhsSrc;
  } else {
    src1 = &rhsSrc;
    src2 = &lhsSrc;
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

  // Dummy TARGET/POINTER argument may alias with a global TARGET/POINTER.
  if (src1->isTargetOrPointer() && src2->isTargetOrPointer() &&
      src1->isData() == src2->isData()) {
    LLVM_DEBUG(llvm::dbgs() << "  aliasing because of target or pointer\n");
    return AliasResult::MayAlias;
  }

  // Box for POINTER component inside an object of a derived type
  // may alias box of a POINTER object, as well as boxes for POINTER
  // components inside two objects of derived types may alias.
  if ((isRecordWithPointerComponent(src1->valueType) &&
       src2->isTargetOrPointer()) ||
      (isRecordWithPointerComponent(src2->valueType) &&
       src1->isTargetOrPointer()) ||
      (isRecordWithPointerComponent(src1->valueType) &&
       isRecordWithPointerComponent(src2->valueType))) {
    LLVM_DEBUG(llvm::dbgs() << "  aliasing because of pointer components\n");
    return AliasResult::MayAlias;
  }

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

AliasAnalysis::Source AliasAnalysis::getSource(mlir::Value v,
                                               bool getInstantiationPoint) {
  auto *defOp = v.getDefiningOp();
  SourceKind type{SourceKind::Unknown};
  mlir::Type ty;
  bool breakFromLoop{false};
  bool approximateSource{false};
  bool followBoxData{mlir::isa<fir::BaseBoxType>(v.getType())};
  bool isBoxRef{fir::isa_ref_type(v.getType()) &&
                mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(v.getType()))};
  bool followingData = !isBoxRef;
  mlir::SymbolRefAttr global;
  Source::Attributes attributes;
  mlir::Value instantiationPoint;
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
        .Case<fir::BoxAddrOp>([&](auto op) {
          v = op->getOperand(0);
          defOp = v.getDefiningOp();
          if (mlir::isa<fir::BaseBoxType>(v.getType()))
            followBoxData = true;
        })
        .Case<fir::ArrayCoorOp, fir::CoordinateOp>([&](auto op) {
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
          if (getInstantiationPoint) {
            // Fetch only the innermost instantiation point.
            if (!instantiationPoint)
              instantiationPoint = op->getResult(0);

            if (op.getDummyScope()) {
              // Do not track past DeclareOp that has the dummy_scope
              // operand. This DeclareOp is known to represent
              // a dummy argument for some runtime instantiation
              // of a procedure.
              type = SourceKind::Argument;
              breakFromLoop = true;
              return;
            }
          }
          // TODO: Look for the fortran attributes present on the operation
          // Track further through the operand
          v = op.getMemref();
          defOp = v.getDefiningOp();
        })
        .Case<hlfir::DesignateOp>([&](auto op) {
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
  if (!defOp && type == SourceKind::Unknown)
    // Check if the memory source is coming through a dummy argument.
    if (isDummyArgument(v)) {
      type = SourceKind::Argument;
      ty = v.getType();
      if (fir::valueHasFirAttribute(v, fir::getTargetAttrName()))
        attributes.set(Attribute::Target);

      if (isPointerReference(ty))
        attributes.set(Attribute::Pointer);
    }

  if (type == SourceKind::Global) {
    return {{global, instantiationPoint, followingData},
            type,
            ty,
            attributes,
            approximateSource};
  }
  return {{v, instantiationPoint, followingData},
          type,
          ty,
          attributes,
          approximateSource};
}

} // namespace fir
