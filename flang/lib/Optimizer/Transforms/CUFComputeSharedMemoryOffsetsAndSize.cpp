//===-- CUFComputeSharedMemoryOffsetsAndSize.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/Target.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Runtime/CUDA/registration.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFCOMPUTESHAREDMEMORYOFFSETSANDSIZE
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace Fortran::runtime::cuda;

namespace {

struct CUFComputeSharedMemoryOffsetsAndSize
    : public fir::impl::CUFComputeSharedMemoryOffsetsAndSizeBase<
          CUFComputeSharedMemoryOffsetsAndSize> {

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::SymbolTable symTab(mod);
    mlir::OpBuilder opBuilder{mod.getBodyRegion()};
    fir::FirOpBuilder builder(opBuilder, mod);
    fir::KindMapping kindMap{fir::getKindMapping(mod)};
    std::optional<mlir::DataLayout> dl =
        fir::support::getOrSetMLIRDataLayout(mod, /*allowDefaultLayout=*/false);
    if (!dl) {
      mlir::emitError(mod.getLoc(),
                      "data layout attribute is required to perform " +
                          getName() + "pass");
    }

    auto gpuMod = cuf::getOrCreateGPUModule(mod, symTab);
    mlir::Type i8Ty = builder.getI8Type();
    mlir::Type i32Ty = builder.getI32Type();
    mlir::Type idxTy = builder.getIndexType();
    for (auto funcOp : gpuMod.getOps<mlir::gpu::GPUFuncOp>()) {
      unsigned nbDynamicSharedVariables = 0;
      unsigned nbStaticSharedVariables = 0;
      uint64_t sharedMemSize = 0;
      unsigned short alignment = 0;
      mlir::Value crtDynOffset;

      // Go over each shared memory operation and compute their start offset and
      // the size and alignment of the global to be generated if all variables
      // are static. If this is dynamic shared memory, then only the alignment
      // is computed.
      for (auto sharedOp : funcOp.getOps<cuf::SharedMemoryOp>()) {
        mlir::Location loc = sharedOp.getLoc();
        builder.setInsertionPoint(sharedOp);
        if (fir::hasDynamicSize(sharedOp.getInType())) {
          mlir::Type ty = sharedOp.getInType();
          if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
            ty = seqTy.getEleTy();
          unsigned short align = dl->getTypeABIAlignment(ty);
          alignment = std::max(alignment, align);
          uint64_t tySize = dl->getTypeSize(ty);
          ++nbDynamicSharedVariables;
          if (crtDynOffset) {
            sharedOp.getOffsetMutable().assign(
                builder.createConvert(loc, i32Ty, crtDynOffset));
          } else {
            mlir::Value zero = builder.createIntegerConstant(loc, i32Ty, 0);
            sharedOp.getOffsetMutable().assign(zero);
          }

          mlir::Value dynSize =
              builder.createIntegerConstant(loc, idxTy, tySize);
          for (auto extent : sharedOp.getShape())
            dynSize =
                mlir::arith::MulIOp::create(builder, loc, dynSize, extent);
          if (crtDynOffset)
            crtDynOffset = mlir::arith::AddIOp::create(builder, loc,
                                                       crtDynOffset, dynSize);
          else
            crtDynOffset = dynSize;

          continue;
        }
        auto [size, align] = fir::getTypeSizeAndAlignmentOrCrash(
            sharedOp.getLoc(), sharedOp.getInType(), *dl, kindMap);
        ++nbStaticSharedVariables;
        mlir::Value offset = builder.createIntegerConstant(
            loc, i32Ty, llvm::alignTo(sharedMemSize, align));
        sharedOp.getOffsetMutable().assign(offset);
        sharedMemSize =
            llvm::alignTo(sharedMemSize, align) + llvm::alignTo(size, align);
        alignment = std::max(alignment, align);
      }

      if (nbDynamicSharedVariables == 0 && nbStaticSharedVariables == 0)
        continue;

      if (nbDynamicSharedVariables > 0 && nbStaticSharedVariables > 0)
        mlir::emitError(
            funcOp.getLoc(),
            "static and dynamic shared variables in a single kernel");

      mlir::DenseElementsAttr init = {};
      if (sharedMemSize > 0) {
        auto vecTy = mlir::VectorType::get(sharedMemSize, i8Ty);
        mlir::Attribute zero = mlir::IntegerAttr::get(i8Ty, 0);
        init = mlir::DenseElementsAttr::get(vecTy, llvm::ArrayRef(zero));
      }

      // Create the shared memory global where each shared variable will point
      // to.
      auto sharedMemType = fir::SequenceType::get(sharedMemSize, i8Ty);
      std::string sharedMemGlobalName =
          (funcOp.getName() + llvm::Twine(cudaSharedMemSuffix)).str();
      mlir::StringAttr linkage = builder.createInternalLinkage();
      builder.setInsertionPointToEnd(gpuMod.getBody());
      llvm::SmallVector<mlir::NamedAttribute> attrs;
      auto globalOpName = mlir::OperationName(fir::GlobalOp::getOperationName(),
                                              gpuMod.getContext());
      attrs.push_back(mlir::NamedAttribute(
          fir::GlobalOp::getDataAttrAttrName(globalOpName),
          cuf::DataAttributeAttr::get(gpuMod.getContext(),
                                      cuf::DataAttribute::Shared)));
      auto sharedMem = fir::GlobalOp::create(
          builder, funcOp.getLoc(), sharedMemGlobalName, false, false,
          sharedMemType, init, linkage, attrs);
      sharedMem.setAlignment(alignment);
    }
  }
};

} // end anonymous namespace
