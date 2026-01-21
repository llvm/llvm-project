//===-- CUFOpConversion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/CUFOpConversion.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/Runtime/CUDA/Descriptor.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/CUDA/allocatable.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/CUDA/memory.h"
#include "flang/Runtime/CUDA/pointer.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include "flang/Support/Fortran.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CUFOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;
using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

namespace {

static bool inDeviceContext(mlir::Operation *op) {
  if (op->getParentOfType<cuf::KernelOp>())
    return true;
  if (op->getParentOfType<mlir::acc::OffloadRegionOpInterface>())
    return true;
  if (auto funcOp = op->getParentOfType<mlir::gpu::GPUFuncOp>())
    return true;
  if (auto funcOp = op->getParentOfType<mlir::func::FuncOp>()) {
    if (auto cudaProcAttr =
            funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                cuf::getProcAttrName())) {
      return cudaProcAttr.getValue() != cuf::ProcAttribute::Host &&
             cudaProcAttr.getValue() != cuf::ProcAttribute::HostDevice;
    }
  }
  return false;
}

static mlir::Value createConvertOp(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Type toTy,
                                   mlir::Value val) {
  if (val.getType() != toTy)
    return fir::ConvertOp::create(rewriter, loc, toTy, val);
  return val;
}

struct DeclareOpConversion : public mlir::OpRewritePattern<fir::DeclareOp> {
  using OpRewritePattern::OpRewritePattern;

  DeclareOpConversion(mlir::MLIRContext *context,
                      const mlir::SymbolTable &symtab)
      : OpRewritePattern(context), symTab{symtab} {}

  mlir::LogicalResult
  matchAndRewrite(fir::DeclareOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getResult().getUsers().empty())
      return success();
    if (auto addrOfOp = op.getMemref().getDefiningOp<fir::AddrOfOp>()) {
      if (inDeviceContext(addrOfOp)) {
        return failure();
      }
      if (auto global = symTab.lookup<fir::GlobalOp>(
              addrOfOp.getSymbol().getRootReference().getValue())) {
        if (cuf::isRegisteredDeviceGlobal(global)) {
          rewriter.setInsertionPointAfter(addrOfOp);
          mlir::Value devAddr = cuf::DeviceAddressOp::create(
              rewriter, op.getLoc(), addrOfOp.getType(), addrOfOp.getSymbol());
          rewriter.startOpModification(op);
          op.getMemrefMutable().assign(devAddr);
          rewriter.finalizeOpModification(op);
          return success();
        }
      }
    }
    return failure();
  }

private:
  const mlir::SymbolTable &symTab;
};

static bool isDstGlobal(cuf::DataTransferOp op) {
  if (auto declareOp = op.getDst().getDefiningOp<fir::DeclareOp>())
    if (declareOp.getMemref().getDefiningOp<fir::AddrOfOp>())
      return true;
  if (auto declareOp = op.getDst().getDefiningOp<hlfir::DeclareOp>())
    if (declareOp.getMemref().getDefiningOp<fir::AddrOfOp>())
      return true;
  return false;
}

static mlir::Value getShapeFromDecl(mlir::Value src) {
  if (auto declareOp = src.getDefiningOp<fir::DeclareOp>())
    return declareOp.getShape();
  if (auto declareOp = src.getDefiningOp<hlfir::DeclareOp>())
    return declareOp.getShape();
  return mlir::Value{};
}

static mlir::Value emboxSrc(mlir::PatternRewriter &rewriter,
                            cuf::DataTransferOp op,
                            const mlir::SymbolTable &symtab,
                            mlir::Type dstEleTy = nullptr) {
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, mod);
  mlir::Value addr;
  mlir::Type srcTy = fir::unwrapRefType(op.getSrc().getType());
  if (fir::isa_trivial(srcTy) &&
      mlir::matchPattern(op.getSrc().getDefiningOp(), mlir::m_Constant())) {
    mlir::Value src = op.getSrc();
    if (srcTy.isInteger(1)) {
      // i1 is not a supported type in the descriptor and it is actually coming
      // from a LOGICAL constant. Use the destination type to avoid mismatch.
      assert(dstEleTy && "expect dst element type to be set");
      srcTy = dstEleTy;
      src = createConvertOp(rewriter, loc, srcTy, src);
      addr = builder.createTemporary(loc, srcTy);
      fir::StoreOp::create(builder, loc, src, addr);
    } else {
      if (dstEleTy && fir::isa_trivial(dstEleTy) && srcTy != dstEleTy) {
        // Use dstEleTy and convert to avoid assign mismatch.
        addr = builder.createTemporary(loc, dstEleTy);
        auto conv = fir::ConvertOp::create(builder, loc, dstEleTy, src);
        fir::StoreOp::create(builder, loc, conv, addr);
        srcTy = dstEleTy;
      } else {
        // Put constant in memory if it is not.
        addr = builder.createTemporary(loc, srcTy);
        fir::StoreOp::create(builder, loc, src, addr);
      }
    }
  } else {
    addr = op.getSrc();
  }
  llvm::SmallVector<mlir::Value> lenParams;
  mlir::Type boxTy = fir::BoxType::get(srcTy);
  mlir::Value box =
      builder.createBox(loc, boxTy, addr, getShapeFromDecl(op.getSrc()),
                        /*slice=*/nullptr, lenParams,
                        /*tdesc=*/nullptr);
  mlir::Value src = builder.createTemporary(loc, box.getType());
  fir::StoreOp::create(builder, loc, box, src);
  return src;
}

static mlir::Value emboxDst(mlir::PatternRewriter &rewriter,
                            cuf::DataTransferOp op,
                            const mlir::SymbolTable &symtab) {
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, mod);
  mlir::Type dstTy = fir::unwrapRefType(op.getDst().getType());
  mlir::Value dstAddr = op.getDst();
  mlir::Type dstBoxTy = fir::BoxType::get(dstTy);
  llvm::SmallVector<mlir::Value> lenParams;
  mlir::Value dstBox =
      builder.createBox(loc, dstBoxTy, dstAddr, getShapeFromDecl(op.getDst()),
                        /*slice=*/nullptr, lenParams,
                        /*tdesc=*/nullptr);
  mlir::Value dst = builder.createTemporary(loc, dstBox.getType());
  fir::StoreOp::create(builder, loc, dstBox, dst);
  return dst;
}

struct CUFDataTransferOpConversion
    : public mlir::OpRewritePattern<cuf::DataTransferOp> {
  using OpRewritePattern::OpRewritePattern;

  CUFDataTransferOpConversion(mlir::MLIRContext *context,
                              const mlir::SymbolTable &symtab,
                              mlir::DataLayout *dl,
                              const fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), symtab{symtab}, dl{dl},
        typeConverter{typeConverter} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::DataTransferOp op,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Type srcTy = fir::unwrapRefType(op.getSrc().getType());
    mlir::Type dstTy = fir::unwrapRefType(op.getDst().getType());

    mlir::Location loc = op.getLoc();
    unsigned mode = 0;
    if (op.getTransferKind() == cuf::DataTransferKind::HostDevice) {
      mode = kHostToDevice;
    } else if (op.getTransferKind() == cuf::DataTransferKind::DeviceHost) {
      mode = kDeviceToHost;
    } else if (op.getTransferKind() == cuf::DataTransferKind::DeviceDevice) {
      mode = kDeviceToDevice;
    } else {
      mlir::emitError(loc, "unsupported transfer kind\n");
    }

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    fir::KindMapping kindMap{fir::getKindMapping(mod)};
    mlir::Value modeValue =
        builder.createIntegerConstant(loc, builder.getI32Type(), mode);

    // Convert data transfer without any descriptor.
    if (!mlir::isa<fir::BaseBoxType>(srcTy) &&
        !mlir::isa<fir::BaseBoxType>(dstTy)) {

      if (fir::isa_trivial(srcTy) && !fir::isa_trivial(dstTy)) {
        // Initialization of an array from a scalar value should be implemented
        // via a kernel launch. Use the flang runtime via the Assign function
        // until we have more infrastructure.
        mlir::Type dstEleTy = fir::unwrapInnerType(fir::unwrapRefType(dstTy));
        mlir::Value src = emboxSrc(rewriter, op, symtab, dstEleTy);
        mlir::Value dst = emboxDst(rewriter, op, symtab);
        mlir::func::FuncOp func =
            fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferCstDesc)>(
                loc, builder);
        auto fTy = func.getFunctionType();
        mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
        mlir::Value sourceLine =
            fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
        llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
            builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
        fir::CallOp::create(builder, loc, func, args);
        rewriter.eraseOp(op);
        return mlir::success();
      }

      mlir::Type i64Ty = builder.getI64Type();
      mlir::Value nbElement =
          cuf::computeElementCount(rewriter, loc, op.getShape(), dstTy, i64Ty);
      unsigned width = 0;
      if (fir::isa_derived(fir::unwrapSequenceType(dstTy))) {
        mlir::Type structTy =
            typeConverter->convertType(fir::unwrapSequenceType(dstTy));
        width = dl->getTypeSizeInBits(structTy) / 8;
      } else {
        width = cuf::computeElementByteSize(loc, dstTy, kindMap);
      }
      mlir::Value widthValue = mlir::arith::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getIntegerAttr(i64Ty, width));
      mlir::Value bytes = nbElement ? mlir::arith::MulIOp::create(
                                          rewriter, loc, nbElement, widthValue)
                                    : widthValue;

      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferPtrPtr)>(loc,
                                                                       builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));

      mlir::Value dst = op.getDst();
      mlir::Value src = op.getSrc();
      // Materialize the src if constant.
      if (matchPattern(src.getDefiningOp(), mlir::m_Constant())) {
        mlir::Value temp = builder.createTemporary(loc, srcTy);
        fir::StoreOp::create(builder, loc, src, temp);
        src = temp;
      }
      llvm::SmallVector<mlir::Value> args{
          fir::runtime::createArguments(builder, loc, fTy, dst, src, bytes,
                                        modeValue, sourceFile, sourceLine)};
      fir::CallOp::create(builder, loc, func, args);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    auto materializeBoxIfNeeded = [&](mlir::Value val) -> mlir::Value {
      if (mlir::isa<fir::EmboxOp, fir::ReboxOp>(val.getDefiningOp())) {
        // Materialize the box to memory to be able to call the runtime.
        mlir::Value box = builder.createTemporary(loc, val.getType());
        fir::StoreOp::create(builder, loc, val, box);
        return box;
      }
      if (mlir::isa<fir::BaseBoxType>(val.getType()))
        if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(val.getDefiningOp()))
          return loadOp.getMemref();
      return val;
    };

    // Conversion of data transfer involving at least one descriptor.
    if (auto dstBoxTy = mlir::dyn_cast<fir::BaseBoxType>(dstTy)) {
      // Transfer to a descriptor.
      mlir::func::FuncOp func =
          isDstGlobal(op)
              ? fir::runtime::getRuntimeFunc<mkRTKey(
                    CUFDataTransferGlobalDescDesc)>(loc, builder)
              : fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferDescDesc)>(
                    loc, builder);
      mlir::Value dst = op.getDst();
      mlir::Value src = op.getSrc();
      if (!mlir::isa<fir::BaseBoxType>(srcTy)) {
        mlir::Type dstEleTy = fir::unwrapInnerType(dstBoxTy.getEleTy());
        src = emboxSrc(rewriter, op, symtab, dstEleTy);
        if (fir::isa_trivial(srcTy))
          func = fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferCstDesc)>(
              loc, builder);
      }

      src = materializeBoxIfNeeded(src);
      dst = materializeBoxIfNeeded(dst);

      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
      fir::CallOp::create(builder, loc, func, args);
      rewriter.eraseOp(op);
    } else {
      // Transfer from a descriptor.
      mlir::Value dst = emboxDst(rewriter, op, symtab);
      mlir::Value src = materializeBoxIfNeeded(op.getSrc());

      mlir::func::FuncOp func = fir::runtime::getRuntimeFunc<mkRTKey(
          CUFDataTransferDescDescNoRealloc)>(loc, builder);

      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
      fir::CallOp::create(builder, loc, func, args);
      rewriter.eraseOp(op);
    }
    return mlir::success();
  }

private:
  const mlir::SymbolTable &symtab;
  mlir::DataLayout *dl;
  const fir::LLVMTypeConverter *typeConverter;
};

struct CUFLaunchOpConversion
    : public mlir::OpRewritePattern<cuf::KernelLaunchOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CUFLaunchOpConversion(mlir::MLIRContext *context,
                        const mlir::SymbolTable &symTab)
      : OpRewritePattern(context), symTab{symTab} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::KernelLaunchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto idxTy = mlir::IndexType::get(op.getContext());
    mlir::Value zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerType(32),
        rewriter.getI32IntegerAttr(0));
    auto gridSizeX =
        mlir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getGridX());
    auto gridSizeY =
        mlir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getGridY());
    auto gridSizeZ =
        mlir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getGridZ());
    auto blockSizeX =
        mlir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getBlockX());
    auto blockSizeY =
        mlir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getBlockY());
    auto blockSizeZ =
        mlir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getBlockZ());
    auto kernelName = mlir::SymbolRefAttr::get(
        rewriter.getStringAttr(cudaDeviceModuleName),
        {mlir::SymbolRefAttr::get(
            rewriter.getContext(),
            op.getCallee().getLeafReference().getValue())});
    mlir::Value clusterDimX, clusterDimY, clusterDimZ;
    cuf::ProcAttributeAttr procAttr;
    if (auto funcOp = symTab.lookup<mlir::func::FuncOp>(
            op.getCallee().getLeafReference())) {
      if (auto clusterDimsAttr = funcOp->getAttrOfType<cuf::ClusterDimsAttr>(
              cuf::getClusterDimsAttrName())) {
        clusterDimX = mlir::arith::ConstantIndexOp::create(
            rewriter, loc, clusterDimsAttr.getX().getInt());
        clusterDimY = mlir::arith::ConstantIndexOp::create(
            rewriter, loc, clusterDimsAttr.getY().getInt());
        clusterDimZ = mlir::arith::ConstantIndexOp::create(
            rewriter, loc, clusterDimsAttr.getZ().getInt());
      }
      procAttr =
          funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName());
    }
    llvm::SmallVector<mlir::Value> args;
    for (mlir::Value arg : op.getArgs()) {
      // If the argument is a global descriptor, make sure we pass the device
      // copy of this descriptor and not the host one.
      if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(arg.getType()))) {
        if (auto declareOp =
                mlir::dyn_cast_or_null<fir::DeclareOp>(arg.getDefiningOp())) {
          if (auto addrOfOp = mlir::dyn_cast_or_null<fir::AddrOfOp>(
                  declareOp.getMemref().getDefiningOp())) {
            if (auto global = symTab.lookup<fir::GlobalOp>(
                    addrOfOp.getSymbol().getRootReference().getValue())) {
              if (cuf::isRegisteredDeviceGlobal(global)) {
                arg = cuf::DeviceAddressOp::create(rewriter, op.getLoc(),
                                                   addrOfOp.getType(),
                                                   addrOfOp.getSymbol())
                          .getResult();
              }
            }
          }
        }
      }
      args.push_back(arg);
    }
    mlir::Value dynamicShmemSize = op.getBytes() ? op.getBytes() : zero;
    auto gpuLaunchOp = mlir::gpu::LaunchFuncOp::create(
        rewriter, loc, kernelName,
        mlir::gpu::KernelDim3{gridSizeX, gridSizeY, gridSizeZ},
        mlir::gpu::KernelDim3{blockSizeX, blockSizeY, blockSizeZ},
        dynamicShmemSize, args);
    if (clusterDimX && clusterDimY && clusterDimZ) {
      gpuLaunchOp.getClusterSizeXMutable().assign(clusterDimX);
      gpuLaunchOp.getClusterSizeYMutable().assign(clusterDimY);
      gpuLaunchOp.getClusterSizeZMutable().assign(clusterDimZ);
    }
    if (op.getStream()) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(gpuLaunchOp);
      mlir::Value stream =
          cuf::StreamCastOp::create(rewriter, loc, op.getStream());
      gpuLaunchOp.getAsyncDependenciesMutable().append(stream);
    }
    if (procAttr)
      gpuLaunchOp->setAttr(cuf::getProcAttrName(), procAttr);
    else
      // Set default global attribute of the original was not found.
      gpuLaunchOp->setAttr(cuf::getProcAttrName(),
                           cuf::ProcAttributeAttr::get(
                               op.getContext(), cuf::ProcAttribute::Global));
    rewriter.replaceOp(op, gpuLaunchOp);
    return mlir::success();
  }

private:
  const mlir::SymbolTable &symTab;
};

struct CUFSyncDescriptorOpConversion
    : public mlir::OpRewritePattern<cuf::SyncDescriptorOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::SyncDescriptorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    auto globalOp = mod.lookupSymbol<fir::GlobalOp>(op.getGlobalName());
    if (!globalOp)
      return mlir::failure();

    auto hostAddr = fir::AddrOfOp::create(
        builder, loc, fir::ReferenceType::get(globalOp.getType()),
        op.getGlobalName());
    fir::runtime::cuda::genSyncGlobalDescriptor(builder, loc, hostAddr);
    op.erase();
    return mlir::success();
  }
};

class CUFOpConversion : public fir::impl::CUFOpConversionBase<CUFOpConversion> {
  using CUFOpConversionBase::CUFOpConversionBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);

    mlir::Operation *op = getOperation();
    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();
    mlir::SymbolTable symtab(module);

    std::optional<mlir::DataLayout> dl = fir::support::getOrSetMLIRDataLayout(
        module, /*allowDefaultLayout=*/false);
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                         /*forceUnifiedTBAATree=*/false, *dl);
    target.addLegalDialect<fir::FIROpsDialect, mlir::arith::ArithDialect,
                           mlir::gpu::GPUDialect>();
    target.addLegalOp<cuf::StreamCastOp>();
    target.addLegalOp<cuf::DeviceAddressOp>();
    cuf::populateCUFToFIRConversionPatterns(typeConverter, *dl, symtab,
                                            patterns);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }

    target.addDynamicallyLegalOp<fir::DeclareOp>([&](fir::DeclareOp op) {
      if (op.getResult().getUsers().empty())
        return true;
      if (inDeviceContext(op))
        return true;
      if (auto addrOfOp = op.getMemref().getDefiningOp<fir::AddrOfOp>()) {
        if (auto global = symtab.lookup<fir::GlobalOp>(
                addrOfOp.getSymbol().getRootReference().getValue())) {
          if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(global.getType())))
            return true;
          if (cuf::isRegisteredDeviceGlobal(global))
            return false;
        }
      }
      return true;
    });

    patterns.clear();
    cuf::populateFIRCUFConversionPatterns(symtab, patterns);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace

void cuf::populateCUFToFIRConversionPatterns(
    const fir::LLVMTypeConverter &converter, mlir::DataLayout &dl,
    const mlir::SymbolTable &symtab, mlir::RewritePatternSet &patterns) {
  patterns.insert<CUFSyncDescriptorOpConversion>(patterns.getContext());
  patterns.insert<CUFDataTransferOpConversion>(patterns.getContext(), symtab,
                                               &dl, &converter);
  patterns.insert<CUFLaunchOpConversion>(patterns.getContext(), symtab);
}

void cuf::populateFIRCUFConversionPatterns(const mlir::SymbolTable &symtab,
                                           mlir::RewritePatternSet &patterns) {
  patterns.insert<DeclareOpConversion>(patterns.getContext(), symtab);
}
