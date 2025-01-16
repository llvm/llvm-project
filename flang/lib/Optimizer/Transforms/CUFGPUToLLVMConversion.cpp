//===-- CUFGPUToLLVMConversion.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/CUFGPUToLLVMConversion.h"
#include "flang/Common/Fortran.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Runtime/CUDA/common.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

namespace fir {
#define GEN_PASS_DEF_CUFGPUTOLLVMCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;
using namespace Fortran::runtime;

namespace {

static mlir::Value createKernelArgArray(mlir::Location loc,
                                        mlir::ValueRange operands,
                                        mlir::PatternRewriter &rewriter) {

  auto *ctx = rewriter.getContext();
  llvm::SmallVector<mlir::Type> structTypes(operands.size(), nullptr);

  for (auto [i, arg] : llvm::enumerate(operands))
    structTypes[i] = arg.getType();

  auto structTy = mlir::LLVM::LLVMStructType::getLiteral(ctx, structTypes);
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Type i32Ty = rewriter.getI32Type();
  auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 0));
  auto one = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 1));
  mlir::Value argStruct =
      rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrTy, structTy, one);
  auto size = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getIntegerAttr(i32Ty, structTypes.size()));
  mlir::Value argArray =
      rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrTy, ptrTy, size);

  for (auto [i, arg] : llvm::enumerate(operands)) {
    auto indice = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getIntegerAttr(i32Ty, i));
    mlir::Value structMember = rewriter.create<LLVM::GEPOp>(
        loc, ptrTy, structTy, argStruct,
        mlir::ArrayRef<mlir::Value>({zero, indice}));
    rewriter.create<LLVM::StoreOp>(loc, arg, structMember);
    mlir::Value arrayMember = rewriter.create<LLVM::GEPOp>(
        loc, ptrTy, ptrTy, argArray, mlir::ArrayRef<mlir::Value>({indice}));
    rewriter.create<LLVM::StoreOp>(loc, structMember, arrayMember);
  }
  return argArray;
}

struct GPULaunchKernelConversion
    : public mlir::ConvertOpToLLVMPattern<mlir::gpu::LaunchFuncOp> {
  explicit GPULaunchKernelConversion(
      const fir::LLVMTypeConverter &typeConverter, mlir::PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::gpu::LaunchFuncOp>(typeConverter,
                                                              benefit) {}

  using OpAdaptor = typename mlir::gpu::LaunchFuncOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
    mlir::Value dynamicMemorySize = op.getDynamicSharedMemorySize();
    mlir::Type i32Ty = rewriter.getI32Type();
    if (!dynamicMemorySize)
      dynamicMemorySize = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 0));

    mlir::Value kernelArgs =
        createKernelArgArray(loc, adaptor.getKernelOperands(), rewriter);

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto kernel = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getKernelName());
    mlir::Value kernelPtr;
    if (!kernel) {
      auto funcOp = mod.lookupSymbol<mlir::func::FuncOp>(op.getKernelName());
      if (!funcOp)
        return mlir::failure();
      kernelPtr =
          rewriter.create<LLVM::AddressOfOp>(loc, ptrTy, funcOp.getName());
    } else {
      kernelPtr =
          rewriter.create<LLVM::AddressOfOp>(loc, ptrTy, kernel.getName());
    }

    auto llvmIntPtrType = mlir::IntegerType::get(
        ctx, this->getTypeConverter()->getPointerBitwidth(0));
    auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);

    mlir::Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);

    if (op.hasClusterSize()) {
      auto funcOp = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          RTNAME_STRING(CUFLaunchClusterKernel));
      auto funcTy = mlir::LLVM::LLVMFunctionType::get(
          voidTy,
          {ptrTy, llvmIntPtrType, llvmIntPtrType, llvmIntPtrType,
           llvmIntPtrType, llvmIntPtrType, llvmIntPtrType, llvmIntPtrType,
           llvmIntPtrType, llvmIntPtrType, i32Ty, ptrTy, ptrTy},
          /*isVarArg=*/false);
      auto cufLaunchClusterKernel = mlir::SymbolRefAttr::get(
          mod.getContext(), RTNAME_STRING(CUFLaunchClusterKernel));
      if (!funcOp) {
        mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto launchKernelFuncOp = rewriter.create<mlir::LLVM::LLVMFuncOp>(
            loc, RTNAME_STRING(CUFLaunchClusterKernel), funcTy);
        launchKernelFuncOp.setVisibility(
            mlir::SymbolTable::Visibility::Private);
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, funcTy, cufLaunchClusterKernel,
          mlir::ValueRange{kernelPtr, adaptor.getClusterSizeX(),
                           adaptor.getClusterSizeY(), adaptor.getClusterSizeZ(),
                           adaptor.getGridSizeX(), adaptor.getGridSizeY(),
                           adaptor.getGridSizeZ(), adaptor.getBlockSizeX(),
                           adaptor.getBlockSizeY(), adaptor.getBlockSizeZ(),
                           dynamicMemorySize, kernelArgs, nullPtr});
    } else {
      auto procAttr =
          op->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName());
      bool isGridGlobal =
          procAttr && procAttr.getValue() == cuf::ProcAttribute::GridGlobal;
      llvm::StringRef fctName = isGridGlobal
                                    ? RTNAME_STRING(CUFLaunchCooperativeKernel)
                                    : RTNAME_STRING(CUFLaunchKernel);
      auto funcOp = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(fctName);
      auto funcTy = mlir::LLVM::LLVMFunctionType::get(
          voidTy,
          {ptrTy, llvmIntPtrType, llvmIntPtrType, llvmIntPtrType,
           llvmIntPtrType, llvmIntPtrType, llvmIntPtrType, i32Ty, ptrTy, ptrTy},
          /*isVarArg=*/false);
      auto cufLaunchKernel =
          mlir::SymbolRefAttr::get(mod.getContext(), fctName);
      if (!funcOp) {
        mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto launchKernelFuncOp =
            rewriter.create<mlir::LLVM::LLVMFuncOp>(loc, fctName, funcTy);
        launchKernelFuncOp.setVisibility(
            mlir::SymbolTable::Visibility::Private);
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, funcTy, cufLaunchKernel,
          mlir::ValueRange{kernelPtr, adaptor.getGridSizeX(),
                           adaptor.getGridSizeY(), adaptor.getGridSizeZ(),
                           adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
                           adaptor.getBlockSizeZ(), dynamicMemorySize,
                           kernelArgs, nullPtr});
    }

    return mlir::success();
  }
};

class CUFGPUToLLVMConversion
    : public fir::impl::CUFGPUToLLVMConversionBase<CUFGPUToLLVMConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);

    mlir::Operation *op = getOperation();
    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();

    std::optional<mlir::DataLayout> dl = fir::support::getOrSetMLIRDataLayout(
        module, /*allowDefaultLayout=*/false);
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                         /*forceUnifiedTBAATree=*/false, *dl);
    cuf::populateCUFGPUToLLVMConversionPatterns(typeConverter, patterns);
    target.addIllegalOp<mlir::gpu::LaunchFuncOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF GPU op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace

void cuf::populateCUFGPUToLLVMConversionPatterns(
    const fir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit) {
  patterns.add<GPULaunchKernelConversion>(converter, benefit);
}
