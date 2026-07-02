//===-- EmitMIFGlobalCtors.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/MIFCommon.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/MIF/MIFOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_EMITMIFGLOBALCTORS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;
using namespace Fortran::runtime;

namespace {

class EmitMIFGlobalCtors
    : public fir::impl::EmitMIFGlobalCtorsBase<EmitMIFGlobalCtors> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext *ctx = mod.getContext();
    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(mod.getBody());
    mlir::Location loc = builder.getUnknownLoc();

    llvm::SmallVector<mlir::Attribute> funcs, priorities, data;
    mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::Attribute zeroAttr = mlir::LLVM::ZeroAttr::get(builder.getContext());

    // Setting priority 0 for the initialization of mif.
    if (mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(mifSaveCoarraysAllocName)) {
      funcs.push_back(
          mlir::FlatSymbolRefAttr::get(ctx, mifSaveCoarraysAllocName));
      priorities.push_back(mlir::IntegerAttr::get(i32Ty, 0));
      data.push_back(zeroAttr);
    }

    if (funcs.empty())
      return;

    // We check whether a GlobalCtorsOp already exists to avoid having
    // multiple instances.
    mlir::LLVM::GlobalCtorsOp globalCtors;
    mod.walk([&](mlir::LLVM::GlobalCtorsOp op) { globalCtors = op; });

    if (globalCtors) {
      llvm::SmallVector<mlir::Attribute> mergedFuncs(
          globalCtors.getCtors().begin(), globalCtors.getCtors().end());
      llvm::SmallVector<mlir::Attribute> mergedPriorities(
          globalCtors.getPriorities().begin(),
          globalCtors.getPriorities().end());
      llvm::SmallVector<mlir::Attribute> mergedData(
          globalCtors.getData().begin(), globalCtors.getData().end());

      llvm::SmallDenseSet<llvm::StringRef> existingNames;
      for (auto attr : globalCtors.getCtors())
        existingNames.insert(
            llvm::cast<mlir::FlatSymbolRefAttr>(attr).getValue());

      for (auto [func, priority, d] : llvm::zip(funcs, priorities, data)) {
        auto name = llvm::cast<mlir::FlatSymbolRefAttr>(func).getValue();
        if (!existingNames.contains(name)) {
          mergedFuncs.push_back(func);
          mergedPriorities.push_back(priority);
          mergedData.push_back(d);
        }
      }

      // Deleting the existing GlobalCtorsOp and creating a new one that
      // contains the merged attributes.
      builder.setInsertionPoint(globalCtors);
      mlir::LLVM::GlobalCtorsOp::create(builder, globalCtors.getLoc(),
                                        builder.getArrayAttr(mergedFuncs),
                                        builder.getArrayAttr(mergedPriorities),
                                        builder.getArrayAttr(mergedData));
      globalCtors.erase();
    } else {
      builder.setInsertionPointToEnd(mod.getBody());
      mlir::LLVM::GlobalCtorsOp::create(
          builder, loc, builder.getArrayAttr(funcs),
          builder.getArrayAttr(priorities), builder.getArrayAttr(data));
    }
  }
};

} // namespace
