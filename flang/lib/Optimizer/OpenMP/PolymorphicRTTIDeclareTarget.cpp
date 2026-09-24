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
#include "llvm/ADT/StringMap.h"

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

    // Keep type processing separate from descendant traversal. Ancestors of a
    // mapped type still need their own type descriptors and inherited bindings
    // processed, but walking from an ancestor to all of its descendants would
    // incorrectly select unrelated sibling types.
    enum WorklistFlags : unsigned {
      ProcessType = 1u << 0,
      IncludeDescendants = 1u << 1,
    };

    struct WorkItem {
      llvm::StringRef typeName;
      unsigned flags;
      llvm::StringRef bindingName;
    };

    llvm::SmallVector<WorkItem, 32> worklist;
    llvm::StringMap<unsigned> processed;
    llvm::StringMap<llvm::SmallSet<llvm::StringRef, 4>> processedBindings;
    llvm::SmallVector<mlir::Operation *, 32> scanWorklist;
    llvm::SmallSet<mlir::Operation *, 32> scanned;

    auto enqueueRecordType = [&](fir::RecordType recordType, unsigned flags,
                                 llvm::StringRef bindingName = {}) {
      if (!recordType)
        return;
      llvm::StringRef typeName =
          fir::NameUniquer::dropTypeConversionMarkers(recordType.getName());
      worklist.push_back({typeName, flags, bindingName});
    };

    auto enqueueScanRoot = [&](mlir::Operation *root) {
      if (root && scanned.insert(root).second)
        scanWorklist.push_back(root);
    };

    auto scanOperationForRTTIUses = [&](mlir::Operation *root) {
      root->walk([&](mlir::Operation *op) {
        if (auto typeDesc = mlir::dyn_cast<fir::TypeDescOp>(op)) {
          enqueueRecordType(getRecordTypeFromType(typeDesc.getInType()),
                            ProcessType | IncludeDescendants);
          return;
        }

        if (auto boxTypeDesc = mlir::dyn_cast<fir::BoxTypeDescOp>(op)) {
          enqueueRecordType(getRecordTypeFromPolymorphicBoxType(
                                boxTypeDesc.getBox().getType()),
                            ProcessType | IncludeDescendants);
          return;
        }

        if (auto dispatch = mlir::dyn_cast<fir::DispatchOp>(op)) {
          enqueueRecordType(getRecordTypeFromPolymorphicBoxType(
                                dispatch.getObject().getType()),
                            ProcessType | IncludeDescendants,
                            dispatch.getMethod());
          return;
        }

        if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
          if (std::optional<mlir::SymbolRefAttr> callee = call.getCallee())
            if (mlir::func::FuncOp func =
                    module.lookupSymbol<mlir::func::FuncOp>(
                        callee->getLeafReference())) {
              // A target region may call a helper routine that performs the
              // actual polymorphic dispatch. Scan direct callees as well so
              // their fir.dispatch ops drive binding-specific declare-target
              // marking. The generic OpenMP MarkDeclareTarget pass owns
              // marking target-reachable helper functions themselves.
              enqueueScanRoot(func.getOperation());
            }
          return;
        }
      });
    };

    //  Walk MapInfoOp's to find possible types needing RTTI on device.
    module.walk([&](mlir::omp::MapInfoOp map) {
      enqueueRecordType(getMappedPolymorphicRecordType(map),
                        ProcessType | IncludeDescendants);
    });

    // Walk target regions and explicit/implicit declare-target functions that
    // may reference RTTI and gather it. Data-only maps only require type
    // descriptor data, but dispatches in device code require the corresponding
    // dispatch-table procedures too.
    module.walk([&](mlir::omp::TargetOp target) {
      enqueueScanRoot(target.getOperation());
    });

    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>())
      if (isDeclareTargetDeviceFunction(func))
        enqueueScanRoot(func.getOperation());

    while (!scanWorklist.empty())
      scanOperationForRTTIUses(scanWorklist.pop_back_val());

    while (!worklist.empty()) {
      WorkItem item = worklist.pop_back_val();

      unsigned newFlags = item.flags;
      auto processedIt = processed.find(item.typeName);
      if (processedIt != processed.end()) {
        newFlags &= ~processedIt->second;
        processedIt->second |= newFlags;
      } else {
        processed[item.typeName] = newFlags;
      }

      bool newBinding = false;
      if (!item.bindingName.empty())
        newBinding =
            processedBindings[item.typeName].insert(item.bindingName).second;

      if (!newFlags && !newBinding)
        continue;

      fir::TypeInfoOp typeInfo =
          fir::lookupTypeInfoOp(item.typeName, module, &symbolTable);
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

      if (newBinding && !typeInfo.getDispatchTable().empty()) {
        for (auto dtEntry :
             typeInfo.getDispatchTable().front().getOps<fir::DTEntryOp>()) {
          if (dtEntry.getMethod() != item.bindingName)
            continue;
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
        enqueueRecordType(parentType, ProcessType, item.bindingName);

      // A mapped polymorphic descriptor may dynamically contain any extension
      // of its declared type that is materialized in this translation unit.
      // Mark those descendant RTTI objects too, so dynamic type descriptors
      // have device-side definitions. When a device operation actually performs
      // polymorphic dispatch, propagate only the dispatched binding name to
      // descendants so unused host-only bindings in the same dispatch tables
      // are not pulled into the device image.
      //
      // Ancestors reached from the declared type are needed for parent type
      // descriptors and inherited bindings, but their unrelated descendants are
      // not possible dynamic types of the originally mapped object. Avoid
      // walking descendants from ancestor-only entries so mapping
      // class(child_a_t) does not select sibling child_b_t RTTI or bindings.
      if (!(item.flags & IncludeDescendants))
        continue;

      for (fir::TypeInfoOp candidate : module.getOps<fir::TypeInfoOp>()) {
        if (candidate == typeInfo)
          continue;
        if (std::optional<llvm::StringRef> parentName =
                candidate.getIfParentName()) {
          llvm::StringRef normalizedParentName =
              fir::NameUniquer::dropTypeConversionMarkers(*parentName);
          if (normalizedParentName == normalizedTypeName)
            enqueueRecordType(candidate.getRecordType(),
                              ProcessType | IncludeDescendants,
                              item.bindingName);
        }
      }
    }
  }
};

} // namespace
