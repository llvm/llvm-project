//===-- CUFCommon.cpp - Shared functions between passes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/AllocationPolicy.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

/// Retrieve or create the CUDA Fortran GPU module in the give in \p mod.
mlir::gpu::GPUModuleOp cuf::getOrCreateGPUModule(mlir::ModuleOp mod,
                                                 mlir::SymbolTable &symTab) {
  if (auto gpuMod = symTab.lookup<mlir::gpu::GPUModuleOp>(cudaDeviceModuleName))
    return gpuMod;

  auto *ctx = mod.getContext();
  mod->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
               mlir::UnitAttr::get(ctx));

  mlir::OpBuilder builder(ctx);
  auto gpuMod = mlir::gpu::GPUModuleOp::create(builder, mod.getLoc(),
                                               cudaDeviceModuleName);
  mlir::Block::iterator insertPt(mod.getBodyRegion().front().end());
  symTab.insert(gpuMod, insertPt);
  return gpuMod;
}

bool cuf::isCUDADeviceContext(mlir::Operation *op) {
  if (!op || !op->getParentRegion())
    return false;
  return isCUDADeviceContext(*op->getParentRegion());
}

// Check if the insertion point is currently in a device context. HostDevice
// subprogram are not considered fully device context so it will return false
// for it.
// If the insertion point is inside an OpenACC region op, it is considered
// device context.
bool cuf::isCUDADeviceContext(mlir::Region &region,
                              bool isDoConcurrentOffloadEnabled) {
  if (region.getParentOfType<cuf::KernelOp>())
    return true;
  if (region.getParentOfType<mlir::acc::ComputeRegionOpInterface>())
    return true;
  if (region.getParentOfType<mlir::acc::HostDataOp>())
    return true;
  if (auto funcOp = region.getParentOfType<mlir::func::FuncOp>()) {
    if (auto cudaProcAttr =
            funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                cuf::getProcAttrName())) {
      return cudaProcAttr.getValue() != cuf::ProcAttribute::Host &&
             cudaProcAttr.getValue() != cuf::ProcAttribute::HostDevice;
    }
  }
  if (isDoConcurrentOffloadEnabled &&
      region.getParentOfType<fir::DoConcurrentLoopOp>())
    return true;
  return false;
}

bool cuf::isRegisteredDeviceAttr(std::optional<cuf::DataAttribute> attr) {
  if (attr && (*attr == cuf::DataAttribute::Device ||
               *attr == cuf::DataAttribute::Managed ||
               *attr == cuf::DataAttribute::Constant))
    return true;
  return false;
}

bool cuf::isRegisteredDeviceGlobal(fir::GlobalOp op) {
  if (op.getConstant())
    return false;
  return isRegisteredDeviceAttr(op.getDataAttr());
}

void cuf::genPointerSync(const mlir::Value box, fir::FirOpBuilder &builder) {
  if (auto declareOp = box.getDefiningOp<hlfir::DeclareOp>()) {
    if (auto addrOfOp = declareOp.getMemref().getDefiningOp<fir::AddrOfOp>()) {
      auto mod = addrOfOp->getParentOfType<mlir::ModuleOp>();
      if (auto globalOp =
              mod.lookupSymbol<fir::GlobalOp>(addrOfOp.getSymbol())) {
        if (cuf::isRegisteredDeviceGlobal(globalOp)) {
          cuf::SyncDescriptorOp::create(builder, box.getLoc(),
                                        addrOfOp.getSymbol());
        }
      }
    }
  }
}

int cuf::computeElementByteSize(mlir::Location loc, mlir::Type type,
                                fir::KindMapping &kindMap,
                                bool emitErrorOnFailure) {
  auto eleTy = fir::unwrapSequenceType(type);
  if (auto t{mlir::dyn_cast<mlir::IntegerType>(eleTy)})
    return t.getWidth() / 8;
  if (auto t{mlir::dyn_cast<mlir::FloatType>(eleTy)})
    return t.getWidth() / 8;
  if (auto t{mlir::dyn_cast<fir::LogicalType>(eleTy)})
    return kindMap.getLogicalBitsize(t.getFKind()) / 8;
  if (auto t{mlir::dyn_cast<mlir::ComplexType>(eleTy)}) {
    int elemSize =
        mlir::cast<mlir::FloatType>(t.getElementType()).getWidth() / 8;
    return 2 * elemSize;
  }
  if (auto t{mlir::dyn_cast<fir::CharacterType>(eleTy)})
    return kindMap.getCharacterBitsize(t.getFKind()) / 8;
  if (emitErrorOnFailure)
    mlir::emitError(loc, "unsupported type");
  return 0;
}

mlir::Value cuf::computeElementCount(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::Value shapeOperand,
                                     mlir::Type seqType,
                                     mlir::Type targetType) {
  if (shapeOperand) {
    // Dynamic extent - extract from shape operand
    llvm::SmallVector<mlir::Value> extents;
    if (auto shapeOp =
            mlir::dyn_cast<fir::ShapeOp>(shapeOperand.getDefiningOp())) {
      extents = shapeOp.getExtents();
    } else if (auto shapeShiftOp = mlir::dyn_cast<fir::ShapeShiftOp>(
                   shapeOperand.getDefiningOp())) {
      for (auto i : llvm::enumerate(shapeShiftOp.getPairs()))
        if (i.index() & 1)
          extents.push_back(i.value());
    }

    if (extents.empty())
      return mlir::Value();

    // Compute total element count by multiplying all dimensions
    mlir::Value count =
        fir::ConvertOp::create(rewriter, loc, targetType, extents[0]);
    for (unsigned i = 1; i < extents.size(); ++i) {
      auto operand =
          fir::ConvertOp::create(rewriter, loc, targetType, extents[i]);
      count = mlir::arith::MulIOp::create(rewriter, loc, count, operand);
    }
    return count;
  } else {
    // Static extent - use constant array size
    if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(seqType)) {
      mlir::IntegerAttr attr =
          rewriter.getIntegerAttr(targetType, seqTy.getConstantArraySize());
      return mlir::arith::ConstantOp::create(rewriter, loc, targetType, attr);
    }
  }
  return mlir::Value();
}

// Factored out of CUFDeviceFuncTransform.cpp (was isDeviceFunc) so that every
// CUF pass tests "has a device side" the same way.
bool cuf::isDeviceProcedure(mlir::func::FuncOp funcOp) {
  auto procAttr =
      funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName());
  if (!procAttr)
    return false;
  switch (procAttr.getValue()) {
  case cuf::ProcAttribute::Device:
  case cuf::ProcAttribute::Global:
  case cuf::ProcAttribute::GridGlobal:
  case cuf::ProcAttribute::HostDevice:
    return true;
  case cuf::ProcAttribute::Host:
    return false;
  }
  return false;
}

// Factored out of CUFDeviceFuncTransform.cpp so that cuf-duplicate-device-func
// and cuf-transform-device-func agree on what device code is. Compared to the
// original: transitive, follows any symbol reference, skips copied originals.
cuf::DeviceCodeSet cuf::collectDeviceCode(mlir::ModuleOp mod,
                                          mlir::SymbolTable &symTab,
                                          bool rejectDynamicDispatch) {
  cuf::DeviceCodeSet code;
  // Originals that already have a device copy are host code from now on.
  llvm::StringSet<> hasDeviceCopy;
  mod.walk([&](mlir::func::FuncOp funcOp) {
    if (cuf::isDeviceProcedure(funcOp)) {
      code.deviceFuncs.insert(funcOp);
      if (std::optional<llvm::StringRef> original =
              cuf::getDeviceCopyOf(funcOp))
        hasDeviceCopy.insert(*original);
    }
  });

  // Everything device code reaches without a device attribute of its own. A
  // worklist makes this transitive: a plain procedure two calls below a kernel
  // is device code too, and must be optimized as such.
  llvm::SmallVector<mlir::Operation *> worklist;
  auto found = [&](mlir::StringAttr name) {
    auto callee = symTab.lookup<mlir::func::FuncOp>(name);
    if (!callee || mlir::acc::isAccRoutine(callee) ||
        code.deviceFuncs.count(callee))
      return;
    if (code.calledFromDevice.insert(callee) && !callee.isDeclaration())
      worklist.push_back(callee);
  };
  auto scan = [&](mlir::Operation *root) {
    // Every symbol referenced in the body, whatever op carries it: fir.call,
    // fir.address_of, and anything added later. The root's own attributes
    // (such as cuf.device_copy_of) are not references to device code.
    for (mlir::Region &region : root->getRegions())
      if (std::optional<mlir::SymbolTable::UseRange> uses =
              mlir::SymbolTable::getSymbolUses(&region))
        for (const mlir::SymbolTable::SymbolUse &use : *uses)
          found(use.getSymbolRef().getLeafReference());
    // Only the outliner asks for this; before the optimizer fir.dispatch is
    // still present in ordinary host code and must not be rejected.
    if (rejectDynamicDispatch)
      root->walk([](fir::DispatchOp op) {
        TODO(op.getLoc(),
             "type-bound procedure call with dynamic dispatch in device code");
      });
  };
  for (mlir::func::FuncOp funcOp : code.deviceFuncs)
    if (!hasDeviceCopy.contains(funcOp.getSymName()))
      worklist.push_back(funcOp);
  mod.walk([&](cuf::KernelOp kernelOp) { worklist.push_back(kernelOp); });
  while (!worklist.empty())
    scan(worklist.pop_back_val());

  // A device procedure referenced from a binding table keeps a host symbol,
  // or the table fails to verify once lowered to the LLVM dialect.
  for (fir::GlobalOp globalOp : mod.getOps<fir::GlobalOp>()) {
    if (!globalOp.getName().contains(fir::kBindingTableSeparator))
      continue;
    globalOp.walk([&](fir::AddrOfOp addrOfOp) {
      auto funcOp = symTab.lookup<mlir::func::FuncOp>(
          addrOfOp.getSymbol().getLeafReference());
      if (funcOp && code.deviceFuncs.count(funcOp))
        code.keepInModule.insert(funcOp);
    });
  }
  return code;
}

// Shared by both passes: the duplicator points device code at the copies and
// the outliner points it back, and the two must rewrite the same ops.
void cuf::remapProcedureSymbols(
    mlir::Operation *root,
    const llvm::DenseMap<mlir::StringAttr, mlir::FlatSymbolRefAttr> &map) {
  if (map.empty())
    return;
  root->walk([&](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
      if (mlir::SymbolRefAttr callee = call.getCalleeAttr())
        if (auto it = map.find(callee.getLeafReference()); it != map.end())
          call.setCalleeAttr(it->second);
    } else if (auto addrOf = mlir::dyn_cast<fir::AddrOfOp>(op)) {
      if (auto it = map.find(addrOf.getSymbol().getLeafReference());
          it != map.end())
        addrOf.setSymbolAttr(it->second);
    }
  });
}

// Shared by lowering (device and global procedures) and the duplicator (device
// copies), so both record the same device policy.
void cuf::setDeviceAllocationPolicy(mlir::Operation *func) {
  fir::AllocationPolicy policy = fir::getAllocationPolicy(func);
  policy.stackArrays = false;
  fir::setAllocationPolicy(func, policy);
}
