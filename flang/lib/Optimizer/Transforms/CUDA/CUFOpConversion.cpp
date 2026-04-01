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
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/IR/Matchers.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CUFOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace aiir;
using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

namespace {

static bool inDeviceContext(aiir::Operation *op) {
  if (op->getParentOfType<cuf::KernelOp>())
    return true;
  if (op->getParentOfType<aiir::acc::OffloadRegionOpInterface>())
    return true;
  if (auto funcOp = op->getParentOfType<aiir::gpu::GPUFuncOp>())
    return true;
  if (auto funcOp = op->getParentOfType<aiir::func::FuncOp>()) {
    if (auto cudaProcAttr =
            funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                cuf::getProcAttrName())) {
      return cudaProcAttr.getValue() != cuf::ProcAttribute::Host &&
             cudaProcAttr.getValue() != cuf::ProcAttribute::HostDevice;
    }
  }
  return false;
}

static aiir::Value createConvertOp(aiir::PatternRewriter &rewriter,
                                   aiir::Location loc, aiir::Type toTy,
                                   aiir::Value val) {
  if (val.getType() != toTy)
    return fir::ConvertOp::create(rewriter, loc, toTy, val);
  return val;
}

struct DeclareOpConversion : public aiir::OpRewritePattern<fir::DeclareOp> {
  using OpRewritePattern::OpRewritePattern;

  DeclareOpConversion(aiir::AIIRContext *context,
                      const aiir::SymbolTable &symtab)
      : OpRewritePattern(context), symTab{symtab} {}

  aiir::LogicalResult
  matchAndRewrite(fir::DeclareOp op,
                  aiir::PatternRewriter &rewriter) const override {
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
          aiir::Value devAddr = cuf::DeviceAddressOp::create(
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
  const aiir::SymbolTable &symTab;
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

static aiir::Value getShapeFromDecl(aiir::Value src) {
  if (auto declareOp = src.getDefiningOp<fir::DeclareOp>())
    return declareOp.getShape();
  if (auto declareOp = src.getDefiningOp<hlfir::DeclareOp>())
    return declareOp.getShape();
  return aiir::Value{};
}

static aiir::Value emboxSrc(aiir::PatternRewriter &rewriter,
                            cuf::DataTransferOp op,
                            const aiir::SymbolTable &symtab,
                            aiir::Type dstEleTy = nullptr) {
  auto mod = op->getParentOfType<aiir::ModuleOp>();
  aiir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, mod);
  aiir::Value addr;
  aiir::Type srcTy = fir::unwrapRefType(op.getSrc().getType());
  if (fir::isa_trivial(srcTy) &&
      aiir::matchPattern(op.getSrc().getDefiningOp(), aiir::m_Constant())) {
    aiir::Value src = op.getSrc();
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
  llvm::SmallVector<aiir::Value> lenParams;
  aiir::Type boxTy = fir::BoxType::get(srcTy);
  aiir::Value box =
      builder.createBox(loc, boxTy, addr, getShapeFromDecl(op.getSrc()),
                        /*slice=*/nullptr, lenParams,
                        /*tdesc=*/nullptr);
  aiir::Value src = builder.createTemporary(loc, box.getType());
  fir::StoreOp::create(builder, loc, box, src);
  return src;
}

static aiir::Value emboxDst(aiir::PatternRewriter &rewriter,
                            cuf::DataTransferOp op,
                            const aiir::SymbolTable &symtab) {
  auto mod = op->getParentOfType<aiir::ModuleOp>();
  aiir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, mod);
  aiir::Type dstTy = fir::unwrapRefType(op.getDst().getType());
  aiir::Value dstAddr = op.getDst();
  aiir::Type dstBoxTy = fir::BoxType::get(dstTy);
  llvm::SmallVector<aiir::Value> lenParams;
  aiir::Value dstBox =
      builder.createBox(loc, dstBoxTy, dstAddr, getShapeFromDecl(op.getDst()),
                        /*slice=*/nullptr, lenParams,
                        /*tdesc=*/nullptr);
  aiir::Value dst = builder.createTemporary(loc, dstBox.getType());
  fir::StoreOp::create(builder, loc, dstBox, dst);
  return dst;
}

struct CUFDataTransferOpConversion
    : public aiir::OpRewritePattern<cuf::DataTransferOp> {
  using OpRewritePattern::OpRewritePattern;

  CUFDataTransferOpConversion(aiir::AIIRContext *context,
                              const aiir::SymbolTable &symtab,
                              aiir::DataLayout *dl,
                              const fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), symtab{symtab}, dl{dl},
        typeConverter{typeConverter} {}

  aiir::LogicalResult
  matchAndRewrite(cuf::DataTransferOp op,
                  aiir::PatternRewriter &rewriter) const override {

    aiir::Type srcTy = fir::unwrapRefType(op.getSrc().getType());
    aiir::Type dstTy = fir::unwrapRefType(op.getDst().getType());

    aiir::Location loc = op.getLoc();
    unsigned mode = 0;
    if (op.getTransferKind() == cuf::DataTransferKind::HostDevice) {
      mode = kHostToDevice;
    } else if (op.getTransferKind() == cuf::DataTransferKind::DeviceHost) {
      mode = kDeviceToHost;
    } else if (op.getTransferKind() == cuf::DataTransferKind::DeviceDevice) {
      mode = kDeviceToDevice;
    } else {
      aiir::emitError(loc, "unsupported transfer kind\n");
    }

    auto mod = op->getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    fir::KindMapping kindMap{fir::getKindMapping(mod)};
    aiir::Value modeValue =
        builder.createIntegerConstant(loc, builder.getI32Type(), mode);

    // Convert data transfer without any descriptor.
    if (!aiir::isa<fir::BaseBoxType>(srcTy) &&
        !aiir::isa<fir::BaseBoxType>(dstTy)) {

      if (fir::isa_trivial(srcTy) && !fir::isa_trivial(dstTy)) {
        // Initialization of an array from a scalar value should be implemented
        // via a kernel launch. Use the flang runtime via the Assign function
        // until we have more infrastructure.
        aiir::Type dstEleTy = fir::unwrapInnerType(fir::unwrapRefType(dstTy));
        aiir::Value src = emboxSrc(rewriter, op, symtab, dstEleTy);
        aiir::Value dst = emboxDst(rewriter, op, symtab);
        aiir::func::FuncOp func =
            fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferCstDesc)>(
                loc, builder);
        auto fTy = func.getFunctionType();
        aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
        aiir::Value sourceLine =
            fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
        llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
            builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
        fir::CallOp::create(builder, loc, func, args);
        rewriter.eraseOp(op);
        return aiir::success();
      }

      aiir::Type i64Ty = builder.getI64Type();
      aiir::Value nbElement =
          cuf::computeElementCount(rewriter, loc, op.getShape(), dstTy, i64Ty);
      unsigned width = 0;
      if (fir::isa_derived(fir::unwrapSequenceType(dstTy))) {
        aiir::Type structTy =
            typeConverter->convertType(fir::unwrapSequenceType(dstTy));
        width = dl->getTypeSizeInBits(structTy) / 8;
      } else {
        width = cuf::computeElementByteSize(loc, dstTy, kindMap);
      }
      aiir::Value widthValue = aiir::arith::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getIntegerAttr(i64Ty, width));
      aiir::Value bytes = nbElement ? aiir::arith::MulIOp::create(
                                          rewriter, loc, nbElement, widthValue)
                                    : widthValue;

      aiir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferPtrPtr)>(loc,
                                                                       builder);
      auto fTy = func.getFunctionType();
      aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      aiir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));

      aiir::Value dst = op.getDst();
      aiir::Value src = op.getSrc();
      // Materialize the src if constant.
      if (matchPattern(src.getDefiningOp(), aiir::m_Constant())) {
        aiir::Value temp = builder.createTemporary(loc, srcTy);
        fir::StoreOp::create(builder, loc, src, temp);
        src = temp;
      }
      llvm::SmallVector<aiir::Value> args{
          fir::runtime::createArguments(builder, loc, fTy, dst, src, bytes,
                                        modeValue, sourceFile, sourceLine)};
      fir::CallOp::create(builder, loc, func, args);
      rewriter.eraseOp(op);
      return aiir::success();
    }

    auto materializeBoxIfNeeded = [&](aiir::Value val) -> aiir::Value {
      if (aiir::isa<fir::EmboxOp, fir::ReboxOp>(val.getDefiningOp())) {
        // Materialize the box to memory to be able to call the runtime.
        aiir::Value box = builder.createTemporary(loc, val.getType());
        fir::StoreOp::create(builder, loc, val, box);
        return box;
      }
      if (aiir::isa<fir::BaseBoxType>(val.getType()))
        if (auto loadOp = aiir::dyn_cast<fir::LoadOp>(val.getDefiningOp()))
          return loadOp.getMemref();
      return val;
    };

    // Conversion of data transfer involving at least one descriptor.
    if (auto dstBoxTy = aiir::dyn_cast<fir::BaseBoxType>(dstTy)) {
      // Transfer to a descriptor.
      aiir::func::FuncOp func =
          isDstGlobal(op)
              ? fir::runtime::getRuntimeFunc<mkRTKey(
                    CUFDataTransferGlobalDescDesc)>(loc, builder)
              : fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferDescDesc)>(
                    loc, builder);
      aiir::Value dst = op.getDst();
      aiir::Value src = op.getSrc();
      if (!aiir::isa<fir::BaseBoxType>(srcTy)) {
        aiir::Type dstEleTy = fir::unwrapInnerType(dstBoxTy.getEleTy());
        src = emboxSrc(rewriter, op, symtab, dstEleTy);
        if (fir::isa_trivial(srcTy))
          func = fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferCstDesc)>(
              loc, builder);
      }

      src = materializeBoxIfNeeded(src);
      dst = materializeBoxIfNeeded(dst);

      auto fTy = func.getFunctionType();
      aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      aiir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
      llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
      fir::CallOp::create(builder, loc, func, args);
      rewriter.eraseOp(op);
    } else {
      // Transfer from a descriptor.
      aiir::Value dst = emboxDst(rewriter, op, symtab);
      aiir::Value src = materializeBoxIfNeeded(op.getSrc());

      aiir::func::FuncOp func = fir::runtime::getRuntimeFunc<mkRTKey(
          CUFDataTransferDescDescNoRealloc)>(loc, builder);

      auto fTy = func.getFunctionType();
      aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      aiir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
      llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
      fir::CallOp::create(builder, loc, func, args);
      rewriter.eraseOp(op);
    }
    return aiir::success();
  }

private:
  const aiir::SymbolTable &symtab;
  aiir::DataLayout *dl;
  const fir::LLVMTypeConverter *typeConverter;
};

struct CUFLaunchOpConversion
    : public aiir::OpRewritePattern<cuf::KernelLaunchOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CUFLaunchOpConversion(aiir::AIIRContext *context,
                        const aiir::SymbolTable &symTab)
      : OpRewritePattern(context), symTab{symTab} {}

  aiir::LogicalResult
  matchAndRewrite(cuf::KernelLaunchOp op,
                  aiir::PatternRewriter &rewriter) const override {
    aiir::Location loc = op.getLoc();
    auto idxTy = aiir::IndexType::get(op.getContext());
    aiir::Value zero = aiir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerType(32),
        rewriter.getI32IntegerAttr(0));
    auto gridSizeX =
        aiir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getGridX());
    auto gridSizeY =
        aiir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getGridY());
    auto gridSizeZ =
        aiir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getGridZ());
    auto blockSizeX =
        aiir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getBlockX());
    auto blockSizeY =
        aiir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getBlockY());
    auto blockSizeZ =
        aiir::arith::IndexCastOp::create(rewriter, loc, idxTy, op.getBlockZ());
    auto kernelName = aiir::SymbolRefAttr::get(
        rewriter.getStringAttr(cudaDeviceModuleName),
        {aiir::SymbolRefAttr::get(
            rewriter.getContext(),
            op.getCallee().getLeafReference().getValue())});
    aiir::Value clusterDimX, clusterDimY, clusterDimZ;
    cuf::ProcAttributeAttr procAttr;
    if (auto funcOp = symTab.lookup<aiir::func::FuncOp>(
            op.getCallee().getLeafReference())) {
      if (auto clusterDimsAttr = funcOp->getAttrOfType<cuf::ClusterDimsAttr>(
              cuf::getClusterDimsAttrName())) {
        clusterDimX = aiir::arith::ConstantIndexOp::create(
            rewriter, loc, clusterDimsAttr.getX().getInt());
        clusterDimY = aiir::arith::ConstantIndexOp::create(
            rewriter, loc, clusterDimsAttr.getY().getInt());
        clusterDimZ = aiir::arith::ConstantIndexOp::create(
            rewriter, loc, clusterDimsAttr.getZ().getInt());
      }
      procAttr =
          funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName());
    }
    llvm::SmallVector<aiir::Value> args;
    for (aiir::Value arg : op.getArgs()) {
      // If the argument is a global descriptor, make sure we pass the device
      // copy of this descriptor and not the host one.
      if (aiir::isa<fir::BaseBoxType>(fir::unwrapRefType(arg.getType()))) {
        if (auto declareOp =
                aiir::dyn_cast_or_null<fir::DeclareOp>(arg.getDefiningOp())) {
          if (auto addrOfOp = aiir::dyn_cast_or_null<fir::AddrOfOp>(
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
    aiir::Value dynamicShmemSize = op.getBytes() ? op.getBytes() : zero;
    auto gpuLaunchOp = aiir::gpu::LaunchFuncOp::create(
        rewriter, loc, kernelName,
        aiir::gpu::KernelDim3{gridSizeX, gridSizeY, gridSizeZ},
        aiir::gpu::KernelDim3{blockSizeX, blockSizeY, blockSizeZ},
        dynamicShmemSize, args);
    if (clusterDimX && clusterDimY && clusterDimZ) {
      gpuLaunchOp.getClusterSizeXMutable().assign(clusterDimX);
      gpuLaunchOp.getClusterSizeYMutable().assign(clusterDimY);
      gpuLaunchOp.getClusterSizeZMutable().assign(clusterDimZ);
    }
    if (op.getStream()) {
      aiir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(gpuLaunchOp);
      aiir::Value stream =
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
    return aiir::success();
  }

private:
  const aiir::SymbolTable &symTab;
};

struct CUFSyncDescriptorOpConversion
    : public aiir::OpRewritePattern<cuf::SyncDescriptorOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(cuf::SyncDescriptorOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    auto globalOp = mod.lookupSymbol<fir::GlobalOp>(op.getGlobalName());
    if (!globalOp)
      return aiir::failure();

    auto hostAddr = fir::AddrOfOp::create(
        builder, loc, fir::ReferenceType::get(globalOp.getType()),
        op.getGlobalName());
    fir::runtime::cuda::genSyncGlobalDescriptor(builder, loc, hostAddr);
    op.erase();
    return aiir::success();
  }
};

class CUFOpConversion : public fir::impl::CUFOpConversionBase<CUFOpConversion> {
  using CUFOpConversionBase::CUFOpConversionBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    aiir::RewritePatternSet patterns(ctx);
    aiir::ConversionTarget target(*ctx);

    aiir::Operation *op = getOperation();
    aiir::ModuleOp module = aiir::dyn_cast<aiir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();
    aiir::SymbolTable symtab(module);

    std::optional<aiir::DataLayout> dl = fir::support::getOrSetAIIRDataLayout(
        module, /*allowDefaultLayout=*/false);
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                         /*forceUnifiedTBAATree=*/false, *dl);
    target.addLegalDialect<fir::FIROpsDialect, aiir::arith::ArithDialect,
                           aiir::gpu::GPUDialect>();
    target.addLegalOp<cuf::StreamCastOp>();
    target.addLegalOp<cuf::DeviceAddressOp>();
    cuf::populateCUFToFIRConversionPatterns(typeConverter, *dl, symtab,
                                            patterns);
    if (aiir::failed(aiir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(ctx),
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
          if (aiir::isa<fir::BaseBoxType>(fir::unwrapRefType(global.getType())))
            return true;
          if (cuf::isRegisteredDeviceGlobal(global))
            return false;
        }
      }
      return true;
    });

    patterns.clear();
    cuf::populateFIRCUFConversionPatterns(symtab, patterns);
    if (aiir::failed(aiir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace

void cuf::populateCUFToFIRConversionPatterns(
    const fir::LLVMTypeConverter &converter, aiir::DataLayout &dl,
    const aiir::SymbolTable &symtab, aiir::RewritePatternSet &patterns) {
  patterns.insert<CUFSyncDescriptorOpConversion>(patterns.getContext());
  patterns.insert<CUFDataTransferOpConversion>(patterns.getContext(), symtab,
                                               &dl, &converter);
  patterns.insert<CUFLaunchOpConversion>(patterns.getContext(), symtab);
}

void cuf::populateFIRCUFConversionPatterns(const aiir::SymbolTable &symtab,
                                           aiir::RewritePatternSet &patterns) {
  patterns.insert<DeclareOpConversion>(patterns.getContext(), symtab);
}
