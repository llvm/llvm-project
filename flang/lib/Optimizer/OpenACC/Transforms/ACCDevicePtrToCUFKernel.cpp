//===- ACCDevicePtrToCUFKernel.cpp  ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A CUDA Fortran kernel launched inside an OpenACC data
// region must receive the device address of any host variable that OpenACC has
// made present, not the host address. This pass wraps each
// cuf.kernel_launch in an acc.host_data construct with acc.use_device operands
// for the mapped host variables, and rebuilds the launch's argument addressing
// on top of the use_device result. The host_data/use_device lowering then
// materializes the present-table device pointer, and any array-section
// addressing is recomputed on the device pointer.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
namespace acc {
#define GEN_PASS_DEF_ACCDEVICEPTRTOCUFKERNEL
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace acc
} // namespace fir

using namespace mlir;

namespace {

/// Walk down an addressing chain to the underlying variable that OpenACC maps
/// as a data-clause pointer
static Value getMappedVar(Value value) {
  while (true) {
    Operation *def = value.getDefiningOp();
    // Stop at the variable OpenACC maps as the data-clause pointer.
    if (isa_and_nonnull<fir::DeclareOp, hlfir::DeclareOp>(def))
      return value;
    if (auto view = dyn_cast_or_null<fir::FortranObjectViewOpInterface>(def)) {
      value = view.getViewSource(cast<OpResult>(value));
      continue;
    }
    // TODO: drop this case once (if) fir.emboxchar implements the
    // FortranObjectViewOpInterface.
    if (auto embox = dyn_cast_or_null<fir::EmboxCharOp>(def)) {
      value = embox.getMemref();
      continue;
    }
    // Descriptor-based (allocatable/pointer) variables: the data address is
    // extracted from the descriptor via box_addr(load(<descriptor ref>)). Peel
    // both so the walk reaches the descriptor variable, which is what OpenACC
    // maps as the data-clause varPtr for such variables.
    if (auto load = value.getDefiningOp<fir::LoadOp>()) {
      // Only a load that produces a descriptor is part of the addressing
      // chain; scalar loads are ordinary values, not addressing steps.
      if (mlir::isa<fir::BaseBoxType>(load.getType())) {
        value = load.getMemref();
        continue;
      }
    }
    return {};
  }
}

/// Collects the host variables made present by an
/// OpenACC data directive that dominates `launch`. The result is
/// computed once per launch, then queried per argument.
static llvm::DenseSet<Value>
collectPresentAccVars(cuf::KernelLaunchOp launch, DominanceInfo &domInfo,
                      PostDominanceInfo &postDomInfo) {
  llvm::DenseSet<Value> presentVars;
  for (Value dataClause :
       acc::getDominatingDataClauses(launch, domInfo, postDomInfo))
    if (Value hostVar = acc::getVar(dataClause.getDefiningOp()))
      if (Value mappedVar = getMappedVar(hostVar))
        presentVars.insert(mappedVar);
  return presentVars;
}

class ACCDevicePtrToCUFKernel
    : public fir::acc::impl::ACCDevicePtrToCUFKernelBase<
          ACCDevicePtrToCUFKernel> {
public:
  using fir::acc::impl::ACCDevicePtrToCUFKernelBase<
      ACCDevicePtrToCUFKernel>::ACCDevicePtrToCUFKernelBase;

  void runOnOperation() override {
    llvm::SmallVector<cuf::KernelLaunchOp> launches;
    getOperation().walk(
        [&](cuf::KernelLaunchOp launch) { launches.push_back(launch); });

    for (cuf::KernelLaunchOp launch : launches)
      rewriteLaunch(launch);
  }

private:
  void rewriteLaunch(cuf::KernelLaunchOp launch) {
    // Collect kernel arguments that are references to a host variable made
    // present by an enclosing acc.data region.
    llvm::SmallVector<OpOperand *> deviceOperands;
    llvm::SetVector<Value> deviceArgs;

    // Special case: a boxchar argument is not pointer-like, so acc.use_device
    // cannot take it. Translate its base address and rebuild the boxchar on the
    // device address. Keyed by the launch operand so the general path below is
    // untouched. TODO: remove once fir.emboxchar is a
    // FortranObjectViewOpInterface.
    llvm::DenseMap<OpOperand *, fir::EmboxCharOp> boxCharArgs;

    DominanceInfo domInfo;
    PostDominanceInfo postDomInfo;
    llvm::DenseSet<Value> presentAccVars =
        collectPresentAccVars(launch, domInfo, postDomInfo);

    for (OpOperand &operand : launch.getArgsMutable()) {
      Value arg = operand.get();
      Value mappedVar = getMappedVar(arg);
      // Check if mappedVar is present due to an enclosing OpenACC data region.
      if (!mappedVar || !presentAccVars.contains(mappedVar))
        continue;
      deviceOperands.push_back(&operand);
      if (auto embox = arg.getDefiningOp<fir::EmboxCharOp>()) {
        boxCharArgs[&operand] = embox;
        deviceArgs.insert(embox.getMemref());   // translate the base pointer
      } else {
        deviceArgs.insert(arg);                 // general case
      }
    }

    if (deviceOperands.empty())
      return;

    OpBuilder builder(launch);
    Location loc = launch.getLoc();

    // One acc.use_device per distinct kernel argument, emitted before the
    // launch so it dominates the host_data region.
    llvm::DenseMap<Value, Value> deviceArgsMap;
    llvm::SmallVector<Value> hostDataOperands;
    for (Value arg : deviceArgs) {
      Value deviceVar = acc::UseDeviceOp::create(builder, loc, arg,
                                                 /*structured=*/true,
                                                 /*implicit=*/false)
                            .getAccVar();
      deviceArgsMap[arg] = deviceVar;
      hostDataOperands.push_back(deviceVar);
    }

    // Wrap the launch in an acc.host_data region.
    auto hostData = acc::HostDataOp::create(builder, loc, /*ifCond=*/Value{},
                                            hostDataOperands);
    hostData.setIfPresent(true);
    Block *body = builder.createBlock(&hostData.getRegion());
    builder.setInsertionPointToStart(body);
    Operation *terminator = acc::TerminatorOp::create(builder, loc);
    launch->moveBefore(terminator);

    for (OpOperand *operand : deviceOperands) {
      if (fir::EmboxCharOp embox = boxCharArgs.lookup(operand)) {
        // Char case: rebuild the boxchar on the device base address. Must be
        // emitted before the launch so the rebuilt value dominates its use.
        builder.setInsertionPoint(launch);
        operand->assign(fir::EmboxCharOp::create(
            builder, embox.getLoc(), embox.getType(),
            deviceArgsMap[embox.getMemref()], embox.getLen()));
      } else {
        // General case
        operand->assign(deviceArgsMap[operand->get()]);
      }
    }
  }
};

} // namespace
