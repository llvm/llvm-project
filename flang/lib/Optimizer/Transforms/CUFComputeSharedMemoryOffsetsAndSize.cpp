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
    for (auto funcOp : gpuMod.getOps<mlir::gpu::GPUFuncOp>()) {
      unsigned nbDynamicSharedVariables = 0;
      unsigned nbStaticSharedVariables = 0;
      uint64_t sharedMemSize = 0;
      unsigned short alignment = 0;

      // Go over each shared memory operation and compute their start offset and
      // the size and alignment of the global to be generated if all variables
      // are static. If this is dynamic shared memory, then only the alignment
      // is computed.
      for (auto sharedOp : funcOp.getOps<cuf::SharedMemoryOp>()) {
        if (fir::hasDynamicSize(sharedOp.getInType())) {
          mlir::Type ty = sharedOp.getInType();
          // getTypeSizeAndAlignmentOrCrash will crash trying to compute the
          // size of an array with dynamic size. Just get the alignment to
          // create the global.
          if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
            ty = seqTy.getEleTy();
          unsigned short align = dl->getTypeABIAlignment(ty);
          ++nbDynamicSharedVariables;
          sharedOp.setOffset(0);
          alignment = std::max(alignment, align);
          continue;
        }
        auto [size, align] = fir::getTypeSizeAndAlignmentOrCrash(
            sharedOp.getLoc(), sharedOp.getInType(), *dl, kindMap);
        ++nbStaticSharedVariables;
        sharedOp.setOffset(llvm::alignTo(sharedMemSize, align));
        sharedMemSize =
            llvm::alignTo(sharedMemSize, align) + llvm::alignTo(size, align);
        alignment = std::max(alignment, align);
      }
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
      auto sharedMem = builder.create<fir::GlobalOp>(
          funcOp.getLoc(), sharedMemGlobalName, false, false, sharedMemType,
          init, linkage, attrs);
      sharedMem.setAlignment(alignment);
    }
  }
};

} // end anonymous namespace
