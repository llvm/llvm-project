//===- PolymorphicRTTIDeclareTarget.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// OpenMP 6.1 polymorphic mapping support needs FIR runtime type information
/// (RTTI) to be available in the device image. This pass runs after
/// MapInfoFinalization, inspects finalized map-info and explicit/implicit
/// declare-target functions, and marks only the required RTTI globals and
/// dispatch-table procedures as OpenMP declare target.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/Dialect/OpenMP/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "omp-polymorphic-rtti-declare-target"

namespace flangomp {
#define GEN_PASS_DEF_POLYMORPHICRTTIDECLARETARGETPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

namespace {

static void markDeclareTarget(mlir::Operation *op, bool implicit) {
  auto declareTargetOp = mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(op);
  if (!declareTargetOp)
    return;

  if (mlir::omp::DeclareTargetAttr existing =
          declareTargetOp.getDeclareTarget()) {
    // Preserve explicit user markings.
    if (!existing.getImplicit())
      return;
  }

  // OpenMP dialect verification rejects implicit=true on globals.
  bool effectiveImplicit = implicit && mlir::isa<mlir::func::FuncOp>(op);
  declareTargetOp.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::any,
                                   mlir::omp::DeclareTargetCaptureClause::to,
                                   /*automap=*/false, effectiveImplicit);
}

static fir::RecordType getRecordTypeFromType(mlir::Type type) {
  return mlir::dyn_cast<fir::RecordType>(
      fir::getDerivedType(fir::unwrapRefType(type)));
}

static fir::RecordType getRecordTypeFromPolymorphicBoxType(mlir::Type type) {
  auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(fir::unwrapRefType(type));
  if (!boxTy || !fir::isPolymorphicType(boxTy) || !fir::boxHasAddendum(boxTy))
    return {};
  return getRecordTypeFromType(boxTy);
}

static fir::RecordType
getMappedPolymorphicRecordType(mlir::omp::MapInfoOp map) {
  auto getFromBoxOffset = [](mlir::Value value) -> fir::RecordType {
    auto boxOffset =
        value ? value.getDefiningOp<fir::BoxOffsetOp>() : fir::BoxOffsetOp{};
    if (!boxOffset || boxOffset.getField() != fir::BoxFieldAttr::derived_type)
      return {};
    return getRecordTypeFromPolymorphicBoxType(boxOffset.getBoxRef().getType());
  };

  if (fir::RecordType recTy = getFromBoxOffset(map.getVarPtr()))
    return recTy;
  if (fir::RecordType recTy = getFromBoxOffset(map.getVarPtrPtr()))
    return recTy;

  return getRecordTypeFromPolymorphicBoxType(map.getVarPtr().getType());
}

static bool isDeclareTargetDeviceFunction(mlir::func::FuncOp func) {
  auto declareTargetOp =
      mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(func.getOperation());
  if (!declareTargetOp)
    return false;

  mlir::omp::DeclareTargetAttr attr = declareTargetOp.getDeclareTarget();
  if (!attr)
    return false;

  mlir::omp::DeclareTargetDeviceType deviceType = attr.getDeviceType();
  return deviceType == mlir::omp::DeclareTargetDeviceType::any ||
         deviceType == mlir::omp::DeclareTargetDeviceType::nohost;
}

class PolymorphicRTTIDeclareTargetPass
    : public flangomp::impl::PolymorphicRTTIDeclareTargetPassBase<
          PolymorphicRTTIDeclareTargetPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    if (mlir::omp::getOpenMPVersionAttribute(module, /*fallback=*/0) < 61)
      return;

    mlir::SymbolTable symbolTable(module);
    llvm::SmallSetVector<llvm::StringRef, 32> worklist;
    llvm::SmallSet<llvm::StringRef, 32> processed;

    auto enqueueRecordType = [&](fir::RecordType recordType) {
      if (!recordType)
        return;
      llvm::StringRef typeName =
          fir::NameUniquer::dropTypeConversionMarkers(recordType.getName());
      worklist.insert(typeName);
    };

    auto scanOperationForRTTIUses = [&](mlir::Operation *root) {
      root->walk([&](mlir::Operation *op) {
        if (auto typeDesc = mlir::dyn_cast<fir::TypeDescOp>(op)) {
          enqueueRecordType(getRecordTypeFromType(typeDesc.getInType()));
          return;
        }

        if (auto boxTypeDesc = mlir::dyn_cast<fir::BoxTypeDescOp>(op)) {
          enqueueRecordType(getRecordTypeFromPolymorphicBoxType(
              boxTypeDesc.getBox().getType()));
          return;
        }

        if (auto dispatch = mlir::dyn_cast<fir::DispatchOp>(op)) {
          enqueueRecordType(getRecordTypeFromPolymorphicBoxType(
              dispatch.getObject().getType()));
          return;
        }
      });
    };

    //  Walk MapInfoOp's to find possible types needing RTTI on device.
    module.walk([&](mlir::omp::MapInfoOp map) {
      enqueueRecordType(getMappedPolymorphicRecordType(map));
    });

    // Walk explicit/implicit declare-target functions that may reference RTTI
    // and gather it.
    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>())
      if (isDeclareTargetDeviceFunction(func))
        scanOperationForRTTIUses(func.getOperation());

    while (!worklist.empty()) {
      llvm::StringRef typeName = worklist.pop_back_val();
      if (!processed.insert(typeName).second)
        continue;

      fir::TypeInfoOp typeInfo =
          fir::lookupTypeInfoOp(typeName, module, &symbolTable);
      if (!typeInfo)
        continue;

      // The lookup is tolerant of type-conversion markers. Use the matched
      // TypeInfoOp record type for all follow-up symbol derivation.
      fir::RecordType recordType = typeInfo.getRecordType();
      llvm::StringRef normalizedTypeName =
          fir::NameUniquer::dropTypeConversionMarkers(recordType.getName());

      std::string typeDescName =
          fir::NameUniquer::getTypeDescriptorName(normalizedTypeName);
      if (fir::GlobalOp typeDescGlobal =
              module.lookupSymbol<fir::GlobalOp>(typeDescName))
        markDeclareTarget(typeDescGlobal.getOperation(), /*implicit=*/false);

      if (!typeInfo.getDispatchTable().empty()) {
        for (auto dtEntry :
             typeInfo.getDispatchTable().front().getOps<fir::DTEntryOp>()) {
          auto proc = dtEntry->getAttrOfType<mlir::SymbolRefAttr>(
              fir::DTEntryOp::getProcAttrNameStr());
          if (!proc)
            continue;
          if (mlir::func::FuncOp func = module.lookupSymbol<mlir::func::FuncOp>(
                  proc.getLeafReference()))
            markDeclareTarget(func.getOperation(), /*implicit=*/true);
        }
      }

      if (fir::RecordType parentType = typeInfo.getIfParentType())
        enqueueRecordType(parentType);

      // A mapped polymorphic descriptor may dynamically contain any extension
      // of its declared type that is materialized in this translation unit.
      // Mark those descendant RTTI objects and dispatch tables too, so dynamic
      // type descriptors and overridden bindings have device-side definitions.
      for (fir::TypeInfoOp candidate : module.getOps<fir::TypeInfoOp>()) {
        if (candidate == typeInfo)
          continue;
        if (std::optional<llvm::StringRef> parentName =
                candidate.getIfParentName()) {
          llvm::StringRef normalizedParentName =
              fir::NameUniquer::dropTypeConversionMarkers(*parentName);
          if (normalizedParentName == normalizedTypeName)
            enqueueRecordType(candidate.getRecordType());
        }
      }
    }
  }
};

} // namespace
