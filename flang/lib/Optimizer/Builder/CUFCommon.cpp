//===-- CUFCommon.cpp - Shared functions between passes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"

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
