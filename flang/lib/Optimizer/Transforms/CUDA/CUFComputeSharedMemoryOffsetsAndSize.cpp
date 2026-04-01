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
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Value.h"
#include "aiir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFCOMPUTESHAREDMEMORYOFFSETSANDSIZE
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace Fortran::runtime::cuda;

namespace {

static bool isAssumedSize(aiir::ValueRange shape) {
  if (shape.size() != 1)
    return false;
  if (llvm::isa_and_nonnull<fir::AssumedSizeExtentOp>(shape[0].getDefiningOp()))
    return true;
  return false;
}

static void createSharedMemoryGlobal(fir::FirOpBuilder &builder,
                                     aiir::Location loc, llvm::StringRef prefix,
                                     llvm::StringRef suffix,
                                     aiir::gpu::GPUModuleOp gpuMod,
                                     aiir::Type sharedMemType, unsigned size,
                                     unsigned align, bool isDynamic) {
  std::string sharedMemGlobalName =
      isDynamic ? (prefix + llvm::Twine(cudaSharedMemSuffix)).str()
                : (prefix + llvm::Twine(cudaSharedMemSuffix) + suffix).str();

  aiir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(gpuMod.getBody());

  aiir::StringAttr linkage = isDynamic ? builder.createExternalLinkage()
                                       : builder.createInternalLinkage();
  llvm::SmallVector<aiir::NamedAttribute> attrs;
  auto globalOpName = aiir::OperationName(fir::GlobalOp::getOperationName(),
                                          gpuMod.getContext());
  attrs.push_back(aiir::NamedAttribute(
      fir::GlobalOp::getDataAttrAttrName(globalOpName),
      cuf::DataAttributeAttr::get(gpuMod.getContext(),
                                  cuf::DataAttribute::Shared)));

  aiir::DenseElementsAttr init = {};
  auto sharedMem =
      fir::GlobalOp::create(builder, loc, sharedMemGlobalName, false, false,
                            sharedMemType, init, linkage, attrs);
  sharedMem.setAlignment(align);
}

struct CUFComputeSharedMemoryOffsetsAndSize
    : public fir::impl::CUFComputeSharedMemoryOffsetsAndSizeBase<
          CUFComputeSharedMemoryOffsetsAndSize> {

  void runOnOperation() override {
    aiir::ModuleOp mod = getOperation();
    aiir::SymbolTable symTab(mod);
    aiir::OpBuilder opBuilder{mod.getBodyRegion()};
    fir::FirOpBuilder builder(opBuilder, mod);
    fir::KindMapping kindMap{fir::getKindMapping(mod)};
    std::optional<aiir::DataLayout> dl =
        fir::support::getOrSetAIIRDataLayout(mod, /*allowDefaultLayout=*/false);
    if (!dl) {
      aiir::emitError(mod.getLoc(),
                      "data layout attribute is required to perform " +
                          getName() + "pass");
    }

    auto gpuMod = cuf::getOrCreateGPUModule(mod, symTab);
    aiir::Type i8Ty = builder.getI8Type();
    aiir::Type i32Ty = builder.getI32Type();
    aiir::Type idxTy = builder.getIndexType();
    for (auto funcOp : gpuMod.getOps<aiir::gpu::GPUFuncOp>()) {
      unsigned nbDynamicSharedVariables = 0;
      unsigned nbStaticSharedVariables = 0;
      uint64_t sharedMemSize = 0;
      unsigned short alignment = 0;
      aiir::Value crtDynOffset;

      // Walk all shared memory operations (including those nested inside
      // scf.parallel from reduction lowering) and compute their start offset
      // and the size and alignment of the global to be generated.
      funcOp.walk([&](cuf::SharedMemoryOp sharedOp) {
        aiir::Location loc = sharedOp.getLoc();
        builder.setInsertionPoint(sharedOp);
        if (fir::hasDynamicSize(sharedOp.getInType())) {
          aiir::Type ty = sharedOp.getInType();
          if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(ty))
            ty = seqTy.getEleTy();
          unsigned short align = dl->getTypeABIAlignment(ty);
          alignment = std::max(alignment, align);
          uint64_t tySize = dl->getTypeSize(ty);
          ++nbDynamicSharedVariables;
          if (isAssumedSize(sharedOp.getShape()) || !crtDynOffset) {
            aiir::Value zero = builder.createIntegerConstant(loc, i32Ty, 0);
            sharedOp.getOffsetMutable().assign(zero);
          } else {
            sharedOp.getOffsetMutable().assign(
                builder.createConvert(loc, i32Ty, crtDynOffset));
          }

          aiir::Value dynSize =
              builder.createIntegerConstant(loc, idxTy, tySize);
          for (auto extent : sharedOp.getShape())
            dynSize =
                aiir::arith::MulIOp::create(builder, loc, dynSize, extent);
          if (crtDynOffset)
            crtDynOffset = aiir::arith::AddIOp::create(builder, loc,
                                                       crtDynOffset, dynSize);
          else
            crtDynOffset = dynSize;
        } else {
          // Static shared memory.
          auto [size, align] = fir::getTypeSizeAndAlignmentOrCrash(
              loc, sharedOp.getInType(), *dl, kindMap);
          createSharedMemoryGlobal(
              builder, sharedOp.getLoc(), funcOp.getName(),
              *sharedOp.getBindcName(), gpuMod,
              fir::SequenceType::get(size, i8Ty), size,
              sharedOp.getAlignment() ? *sharedOp.getAlignment() : align,
              /*isDynamic=*/false);
          aiir::Value zero = builder.createIntegerConstant(loc, i32Ty, 0);
          sharedOp.getOffsetMutable().assign(zero);
          if (!sharedOp.getAlignment())
            sharedOp.setAlignment(align);
          sharedOp.setIsStatic(true);
          ++nbStaticSharedVariables;
        }
      });

      if (nbDynamicSharedVariables == 0 && nbStaticSharedVariables == 0)
        continue;

      if (nbDynamicSharedVariables > 0 && nbStaticSharedVariables > 0)
        aiir::emitError(
            funcOp.getLoc(),
            "static and dynamic shared variables in a single kernel");

      if (nbStaticSharedVariables > 0)
        continue;

      auto sharedMemType = fir::SequenceType::get(sharedMemSize, i8Ty);
      createSharedMemoryGlobal(builder, funcOp.getLoc(), funcOp.getName(), "",
                               gpuMod, sharedMemType, sharedMemSize, alignment,
                               /*isDynamic=*/true);
    }
  }
};

} // end anonymous namespace
