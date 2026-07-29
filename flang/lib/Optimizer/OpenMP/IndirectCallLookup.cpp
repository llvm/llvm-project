//===- IndirectCallLookup.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// On a GPU target device, rewrites indirect fir.call ops so the callee (a host
// function address held in a procedure pointer) is resolved to the device
// address via the `__llvm_omp_indirect_call_lookup` runtime function.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace flangomp {
#define GEN_PASS_DEF_INDIRECTCALLLOOKUPPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

/// Runtime function that maps a host function address to the device address.
static constexpr llvm::StringRef indirectCallLookupName =
    "__llvm_omp_indirect_call_lookup";

namespace {
class IndirectCallLookupPass
    : public flangomp::impl::IndirectCallLookupPassBase<IndirectCallLookupPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    auto offloadMod = mlir::dyn_cast<mlir::omp::OffloadModuleInterface>(
        module.getOperation());

    // Only a GPU target device needs host-to-device address translation.
    if (!offloadMod || !offloadMod.getIsTargetDevice() ||
        !offloadMod.getIsGPU())
      return;

    // An indirect fir.call has no callee symbol; operand 0 is the callee value.
    llvm::SmallVector<fir::CallOp> indirectCalls;
    module.walk([&](fir::CallOp call) {
      if (!call.getCallee())
        indirectCalls.push_back(call);
    });
    if (indirectCalls.empty())
      return;

    mlir::MLIRContext *ctx = &getContext();
    mlir::OpBuilder builder(ctx);

    // A function value lowers to a pointer, so an opaque `() -> ()` type matches
    // the runtime function's ptr argument and result.
    auto opaqueFnTy = mlir::FunctionType::get(ctx, {}, {});

    // Declare the runtime lookup function once.
    auto lookupFn =
        module.lookupSymbol<mlir::func::FuncOp>(indirectCallLookupName);
    if (!lookupFn) {
      builder.setInsertionPointToStart(module.getBody());
      lookupFn = mlir::func::FuncOp::create(
          builder, module.getLoc(), indirectCallLookupName,
          mlir::FunctionType::get(ctx, {opaqueFnTy}, {opaqueFnTy}));
      lookupFn.setPrivate();
    }

    for (fir::CallOp call : indirectCalls) {
      builder.setInsertionPoint(call);
      mlir::Location loc = call.getLoc();
      mlir::Value callee = call.getOperand(0);

      // Resolve the host callee to the device address, then call through it.
      mlir::Value hostAddr =
          fir::ConvertOp::create(builder, loc, opaqueFnTy, callee);
      auto lookup = fir::CallOp::create(builder, loc, lookupFn,
                                        mlir::ValueRange{hostAddr});
      mlir::Value deviceCallee = fir::ConvertOp::create(
          builder, loc, callee.getType(), lookup.getResult(0));
      call.setOperand(0, deviceCallee);
    }
  }
};
} // namespace
