//===- ACCDevicePtrToCUFKernel.cpp  --------------------------------------===//
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
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
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
    if (auto convert = value.getDefiningOp<fir::ConvertOp>()) {
      value = convert.getValue();
      continue;
    }
    if (auto coor = value.getDefiningOp<fir::ArrayCoorOp>()) {
      value = coor.getMemref();
      continue;
    }
    if (auto coor = value.getDefiningOp<fir::CoordinateOp>()) {
      value = coor.getRef();
      continue;
    }
    if (auto designate = value.getDefiningOp<hlfir::DesignateOp>()) {
      value = designate.getMemref();
      continue;
    }
    // Descriptor-based (allocatable/pointer) variables: the data address is
    // extracted from the descriptor via box_addr(load(<descriptor ref>)). Peel
    // both so the walk reaches the descriptor variable, which is what OpenACC
    // maps as the data-clause varPtr for such variables.
    if (auto boxAddr = value.getDefiningOp<fir::BoxAddrOp>()) {
      value = boxAddr.getVal();
      continue;
    }
    if (auto load = value.getDefiningOp<fir::LoadOp>()) {
      // Only a load that produces a descriptor is part of the addressing
      // chain; scalar loads are ordinary values, not addressing steps.
      if (mlir::isa<fir::BaseBoxType>(load.getType())) {
        value = load.getMemref();
        continue;
      }
    }
    if (isa_and_nonnull<fir::DeclareOp, hlfir::DeclareOp>(
            value.getDefiningOp()))
      return value;
    return {};
  }
}

/// Checks if mappedVar is present due to an enclosing acc.data region.
static bool isMappedInEnclosingAccData(Value mappedVar,
                                       cuf::KernelLaunchOp launch) {
  if(!mappedVar)
    return false;
  for (auto dataOp = launch->getParentOfType<acc::DataOp>(); dataOp;
       dataOp = dataOp->getParentOfType<acc::DataOp>()) {
    for (Value dataOperand : dataOp.getDataClauseOperands()) {
      if (Value hostVar = acc::getVar(dataOperand.getDefiningOp()))
        if (getMappedVar(hostVar) == mappedVar)
          return true;
    }
  }
  return false;
}

/// Reconstructs the addressing chain that produced `value` from `mappedVar`,
/// substituting `deviceVar` for `mappedVar`. Only addressing ops are cloned;
/// everything else (constants, shapes, ...) is reused as a live-in. New ops are
/// created at `builder`'s current insertion point.
static Value rebuildOnDevice(OpBuilder &builder, Value value, Value mappedVar,
                             Value deviceVar) {
  if (value == mappedVar)
    return deviceVar;

  Operation *def = value.getDefiningOp();
  if (!def || !isa<fir::ConvertOp, fir::ArrayCoorOp, fir::CoordinateOp,
                   hlfir::DesignateOp, fir::BoxAddrOp, fir::LoadOp>(def))
    return value;

  // Mirror getMappedVar: only a descriptor load is part of the addressing
  // chain and must be rebuilt on the device descriptor; any other load is a
  // live-in and is reused as-is.
  if (auto load = dyn_cast<fir::LoadOp>(def))
    if (!mlir::isa<fir::BaseBoxType>(load.getType()))
      return value;

  IRMapping map;
  for (Value operand : def->getOperands())
    map.map(operand, rebuildOnDevice(builder, operand, mappedVar, deviceVar));
  return builder.clone(*def, map)->getResult(0);
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
    struct MappedArg {
      OpOperand *operand;
      Value mappedVar;
    };
    llvm::SmallVector<MappedArg> mappedArgs;
    llvm::SetVector<Value> mappedVars;

    for (OpOperand &operand : launch.getArgsMutable()) {
      Value arg = operand.get();
      if (!fir::isa_ref_type(arg.getType()))
        continue;
      Value mappedVar = getMappedVar(arg);
      if (!isMappedInEnclosingAccData(mappedVar, launch))
        continue;
      mappedArgs.push_back({&operand, mappedVar});
      mappedVars.insert(mappedVar);
    }

    if (mappedArgs.empty())
      return;

    OpBuilder builder(launch);
    Location loc = launch.getLoc();

    // One acc.use_device per distinct mapped variable, emitted before the
    // launch so it dominates the host_data region.
    llvm::DenseMap<Value, Value> deviceVars;
    llvm::SmallVector<Value> dataOperands;
    for (Value mappedVar : mappedVars) {
      Value deviceVar = acc::UseDeviceOp::create(builder, loc, mappedVar,
                                                 /*structured=*/true,
                                                 /*implicit=*/false)
                            .getAccVar();
      deviceVars[mappedVar] = deviceVar;
      dataOperands.push_back(deviceVar);
    }

    // Wrap the launch in an acc.host_data region.
    auto hostData =
        acc::HostDataOp::create(builder, loc, /*ifCond=*/Value{}, dataOperands);
    Block *body = builder.createBlock(&hostData.getRegion());
    builder.setInsertionPointToStart(body);
    Operation *terminator = acc::TerminatorOp::create(builder, loc);
    launch->moveBefore(terminator);

    // Recompute each mapped argument's address on the device pointer.
    builder.setInsertionPoint(launch);
    for (MappedArg &mappedArg : mappedArgs) {
      Value arg = mappedArg.operand->get();
      Value deviceVar = deviceVars[mappedArg.mappedVar];
      mappedArg.operand->assign(
          rebuildOnDevice(builder, arg, mappedArg.mappedVar, deviceVar));
    }
  }
};

} // namespace
