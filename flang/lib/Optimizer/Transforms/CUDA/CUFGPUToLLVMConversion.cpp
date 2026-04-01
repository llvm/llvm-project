//===-- CUFGPUToLLVMConversion.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/CUFGPUToLLVMConversion.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Support/Fortran.h"
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

namespace fir {
#define GEN_PASS_DEF_CUFGPUTOLLVMCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace aiir;
using namespace Fortran::runtime;

namespace {

static aiir::Value createKernelArgArray(aiir::Location loc,
                                        aiir::ValueRange operands,
                                        aiir::PatternRewriter &rewriter) {

  auto *ctx = rewriter.getContext();
  llvm::SmallVector<aiir::Type> structTypes(operands.size(), nullptr);

  for (auto [i, arg] : llvm::enumerate(operands))
    structTypes[i] = arg.getType();

  auto structTy = aiir::LLVM::LLVMStructType::getLiteral(ctx, structTypes);
  auto ptrTy = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  aiir::Type i32Ty = rewriter.getI32Type();
  auto zero = aiir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getIntegerAttr(i32Ty, 0));
  auto one = aiir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getIntegerAttr(i32Ty, 1));
  aiir::Value argStruct =
      aiir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, structTy, one);
  auto size = aiir::LLVM::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getIntegerAttr(i32Ty, structTypes.size()));
  aiir::Value argArray =
      aiir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, size);

  for (auto [i, arg] : llvm::enumerate(operands)) {
    auto indice = aiir::LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getIntegerAttr(i32Ty, i));
    aiir::Value structMember =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy, argStruct,
                            aiir::ArrayRef<aiir::Value>({zero, indice}));
    LLVM::StoreOp::create(rewriter, loc, arg, structMember);
    aiir::Value arrayMember =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrTy, argArray,
                            aiir::ArrayRef<aiir::Value>({indice}));
    LLVM::StoreOp::create(rewriter, loc, structMember, arrayMember);
  }
  return argArray;
}

struct GPULaunchKernelConversion
    : public aiir::ConvertOpToLLVMPattern<aiir::gpu::LaunchFuncOp> {
  explicit GPULaunchKernelConversion(
      const fir::LLVMTypeConverter &typeConverter, aiir::PatternBenefit benefit)
      : aiir::ConvertOpToLLVMPattern<aiir::gpu::LaunchFuncOp>(typeConverter,
                                                              benefit) {}

  using OpAdaptor = typename aiir::gpu::LaunchFuncOp::Adaptor;

  aiir::LogicalResult
  matchAndRewrite(aiir::gpu::LaunchFuncOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    // Only convert gpu.launch_func for CUDA Fortran.
    if (!op.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
            cuf::getProcAttrName()))
      return aiir::failure();

    aiir::Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    aiir::ModuleOp mod = op->getParentOfType<aiir::ModuleOp>();
    aiir::Value dynamicMemorySize = op.getDynamicSharedMemorySize();
    aiir::Type i32Ty = rewriter.getI32Type();
    if (!dynamicMemorySize)
      dynamicMemorySize = aiir::LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 0));

    aiir::Value kernelArgs =
        createKernelArgArray(loc, adaptor.getKernelOperands(), rewriter);

    auto ptrTy = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto kernel = mod.lookupSymbol<aiir::LLVM::LLVMFuncOp>(op.getKernelName());
    aiir::Value kernelPtr;
    if (!kernel) {
      auto funcOp = mod.lookupSymbol<aiir::func::FuncOp>(op.getKernelName());
      if (!funcOp)
        return aiir::failure();
      kernelPtr =
          LLVM::AddressOfOp::create(rewriter, loc, ptrTy, funcOp.getName());
    } else {
      kernelPtr =
          LLVM::AddressOfOp::create(rewriter, loc, ptrTy, kernel.getName());
    }

    auto llvmIntPtrType = aiir::IntegerType::get(
        ctx, this->getTypeConverter()->getPointerBitwidth(0));
    auto voidTy = aiir::LLVM::LLVMVoidType::get(ctx);

    aiir::Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);

    if (op.hasClusterSize()) {
      auto funcOp = mod.lookupSymbol<aiir::LLVM::LLVMFuncOp>(
          RTNAME_STRING(CUFLaunchClusterKernel));
      auto funcTy = aiir::LLVM::LLVMFunctionType::get(
          voidTy,
          {ptrTy, llvmIntPtrType, llvmIntPtrType, llvmIntPtrType,
           llvmIntPtrType, llvmIntPtrType, llvmIntPtrType, llvmIntPtrType,
           llvmIntPtrType, llvmIntPtrType, ptrTy, i32Ty, ptrTy, ptrTy},
          /*isVarArg=*/false);
      auto cufLaunchClusterKernel = aiir::SymbolRefAttr::get(
          mod.getContext(), RTNAME_STRING(CUFLaunchClusterKernel));
      if (!funcOp) {
        aiir::OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto launchKernelFuncOp = aiir::LLVM::LLVMFuncOp::create(
            rewriter, loc, RTNAME_STRING(CUFLaunchClusterKernel), funcTy);
        launchKernelFuncOp.setVisibility(
            aiir::SymbolTable::Visibility::Private);
      }

      aiir::Value stream = nullPtr;
      if (!adaptor.getAsyncDependencies().empty()) {
        if (adaptor.getAsyncDependencies().size() != 1)
          return rewriter.notifyMatchFailure(
              op, "Can only convert with exactly one stream dependency.");
        stream = adaptor.getAsyncDependencies().front();
      }

      aiir::LLVM::CallOp::create(
          rewriter, loc, funcTy, cufLaunchClusterKernel,
          aiir::ValueRange{kernelPtr, adaptor.getClusterSizeX(),
                           adaptor.getClusterSizeY(), adaptor.getClusterSizeZ(),
                           adaptor.getGridSizeX(), adaptor.getGridSizeY(),
                           adaptor.getGridSizeZ(), adaptor.getBlockSizeX(),
                           adaptor.getBlockSizeY(), adaptor.getBlockSizeZ(),
                           stream, dynamicMemorySize, kernelArgs, nullPtr});
      rewriter.eraseOp(op);
    } else {
      auto procAttr =
          op->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName());
      bool isGridGlobal =
          procAttr && procAttr.getValue() == cuf::ProcAttribute::GridGlobal;
      llvm::StringRef fctName = isGridGlobal
                                    ? RTNAME_STRING(CUFLaunchCooperativeKernel)
                                    : RTNAME_STRING(CUFLaunchKernel);
      auto funcOp = mod.lookupSymbol<aiir::LLVM::LLVMFuncOp>(fctName);
      auto funcTy = aiir::LLVM::LLVMFunctionType::get(
          voidTy,
          {ptrTy, llvmIntPtrType, llvmIntPtrType, llvmIntPtrType,
           llvmIntPtrType, llvmIntPtrType, llvmIntPtrType, ptrTy, i32Ty, ptrTy,
           ptrTy},
          /*isVarArg=*/false);
      auto cufLaunchKernel =
          aiir::SymbolRefAttr::get(mod.getContext(), fctName);
      if (!funcOp) {
        aiir::OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto launchKernelFuncOp =
            aiir::LLVM::LLVMFuncOp::create(rewriter, loc, fctName, funcTy);
        launchKernelFuncOp.setVisibility(
            aiir::SymbolTable::Visibility::Private);
      }

      aiir::Value stream = nullPtr;
      if (!adaptor.getAsyncDependencies().empty()) {
        if (adaptor.getAsyncDependencies().size() != 1)
          return rewriter.notifyMatchFailure(
              op, "Can only convert with exactly one stream dependency.");
        stream = adaptor.getAsyncDependencies().front();
      }

      aiir::LLVM::CallOp::create(
          rewriter, loc, funcTy, cufLaunchKernel,
          aiir::ValueRange{kernelPtr, adaptor.getGridSizeX(),
                           adaptor.getGridSizeY(), adaptor.getGridSizeZ(),
                           adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
                           adaptor.getBlockSizeZ(), stream, dynamicMemorySize,
                           kernelArgs, nullPtr});
      rewriter.eraseOp(op);
    }

    return aiir::success();
  }
};

static std::string getFuncName(cuf::SharedMemoryOp op) {
  if (auto gpuFuncOp = op->getParentOfType<aiir::gpu::GPUFuncOp>())
    return gpuFuncOp.getName().str();
  if (auto funcOp = op->getParentOfType<aiir::func::FuncOp>())
    return funcOp.getName().str();
  if (auto llvmFuncOp = op->getParentOfType<aiir::LLVM::LLVMFuncOp>())
    return llvmFuncOp.getSymName().str();
  return "";
}

static aiir::Value createAddressOfOp(aiir::ConversionPatternRewriter &rewriter,
                                     aiir::Location loc,
                                     gpu::GPUModuleOp gpuMod,
                                     std::string &sharedGlobalName) {
  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(
      rewriter.getContext(),
      static_cast<unsigned>(aiir::NVVM::NVVMMemorySpace::Shared));
  if (auto g = gpuMod.lookupSymbol<fir::GlobalOp>(sharedGlobalName))
    return aiir::LLVM::AddressOfOp::create(rewriter, loc, llvmPtrTy,
                                           g.getSymName());
  if (auto g = gpuMod.lookupSymbol<aiir::LLVM::GlobalOp>(sharedGlobalName))
    return aiir::LLVM::AddressOfOp::create(rewriter, loc, llvmPtrTy,
                                           g.getSymName());
  return {};
}

struct CUFSharedMemoryOpConversion
    : public aiir::ConvertOpToLLVMPattern<cuf::SharedMemoryOp> {
  explicit CUFSharedMemoryOpConversion(
      const fir::LLVMTypeConverter &typeConverter, aiir::PatternBenefit benefit)
      : aiir::ConvertOpToLLVMPattern<cuf::SharedMemoryOp>(typeConverter,
                                                          benefit) {}
  using OpAdaptor = typename cuf::SharedMemoryOp::Adaptor;

  aiir::LogicalResult
  matchAndRewrite(cuf::SharedMemoryOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = op->getLoc();
    auto gpuMod = op->getParentOfType<gpu::GPUModuleOp>();

    std::string sharedGlobalName =
        op.getIsStatic()
            ? (getFuncName(op) + llvm::Twine(cudaSharedMemSuffix) +
               *op.getBindcName())
                  .str()
            : (getFuncName(op) + llvm::Twine(cudaSharedMemSuffix)).str();
    aiir::Value sharedGlobalAddr =
        createAddressOfOp(rewriter, loc, gpuMod, sharedGlobalName);

    if (!sharedGlobalAddr)
      aiir::emitError(loc, "Could not find the shared global operation\n");

    auto castPtr = aiir::LLVM::AddrSpaceCastOp::create(
        rewriter, loc, aiir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        sharedGlobalAddr);
    aiir::Type baseType = castPtr->getResultTypes().front();
    aiir::LLVM::GEPArg offsetArg =
        op.getOffset() ? aiir::LLVM::GEPArg(op.getOffset())
                       : aiir::LLVM::GEPArg(static_cast<int32_t>(0));
    llvm::SmallVector<aiir::LLVM::GEPArg> gepArgs = {offsetArg};
    aiir::Value shmemPtr = aiir::LLVM::GEPOp::create(
        rewriter, loc, baseType, rewriter.getI8Type(), castPtr, gepArgs);
    rewriter.replaceOp(op, {shmemPtr});
    return aiir::success();
  }
};

struct CUFStreamCastConversion
    : public aiir::ConvertOpToLLVMPattern<cuf::StreamCastOp> {
  explicit CUFStreamCastConversion(const fir::LLVMTypeConverter &typeConverter,
                                   aiir::PatternBenefit benefit)
      : aiir::ConvertOpToLLVMPattern<cuf::StreamCastOp>(typeConverter,
                                                        benefit) {}
  using OpAdaptor = typename cuf::StreamCastOp::Adaptor;

  aiir::LogicalResult
  matchAndRewrite(cuf::StreamCastOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getStream());
    return aiir::success();
  }
};

class CUFGPUToLLVMConversion
    : public fir::impl::CUFGPUToLLVMConversionBase<CUFGPUToLLVMConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    aiir::RewritePatternSet patterns(ctx);
    aiir::ConversionTarget target(*ctx);

    aiir::Operation *op = getOperation();
    aiir::ModuleOp module = aiir::dyn_cast<aiir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();

    std::optional<aiir::DataLayout> dl = fir::support::getOrSetAIIRDataLayout(
        module, /*allowDefaultLayout=*/false);
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                         /*forceUnifiedTBAATree=*/false, *dl);
    cuf::populateCUFGPUToLLVMConversionPatterns(typeConverter, patterns);

    target.addDynamicallyLegalOp<aiir::gpu::LaunchFuncOp>(
        [&](aiir::gpu::LaunchFuncOp op) {
          if (op.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                  cuf::getProcAttrName()))
            return false;
          return true;
        });

    target.addIllegalOp<cuf::SharedMemoryOp>();
    target.addLegalDialect<aiir::LLVM::LLVMDialect>();
    if (aiir::failed(aiir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(ctx),
                      "error in CUF GPU op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace

void cuf::populateCUFGPUToLLVMConversionPatterns(
    fir::LLVMTypeConverter &converter, aiir::RewritePatternSet &patterns,
    aiir::PatternBenefit benefit) {
  converter.addConversion([&converter](aiir::gpu::AsyncTokenType) -> Type {
    return aiir::LLVM::LLVMPointerType::get(&converter.getContext());
  });
  patterns.add<CUFSharedMemoryOpConversion, GPULaunchKernelConversion,
               CUFStreamCastConversion>(converter, benefit);
}
