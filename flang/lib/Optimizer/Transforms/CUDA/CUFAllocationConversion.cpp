//===-- CUFAllocationConversion.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/CUDA/CUFAllocationConversion.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/CUDA/Descriptor.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Runtime/CUDA/allocatable.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/CUDA/memory.h"
#include "flang/Runtime/CUDA/pointer.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include "flang/Support/Fortran.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/Matchers.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CUFALLOCATIONCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace aiir;
using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

namespace {

template <typename OpTy>
static bool isPinned(OpTy op) {
  if (op.getDataAttr() && *op.getDataAttr() == cuf::DataAttribute::Pinned)
    return true;
  return false;
}

static inline unsigned getMemType(cuf::DataAttribute attr) {
  if (attr == cuf::DataAttribute::Device)
    return kMemTypeDevice;
  if (attr == cuf::DataAttribute::Managed)
    return kMemTypeManaged;
  if (attr == cuf::DataAttribute::Pinned)
    return kMemTypePinned;
  if (attr == cuf::DataAttribute::Unified)
    return kMemTypeUnified;
  llvm_unreachable("unsupported memory type");
}

static bool inDeviceContext(aiir::Operation *op) {
  if (op->getParentOfType<cuf::KernelOp>())
    return true;
  if (auto funcOp = op->getParentOfType<aiir::gpu::GPUFuncOp>())
    return true;
  if (auto funcOp = op->getParentOfType<aiir::gpu::LaunchOp>())
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

template <typename OpTy>
static aiir::LogicalResult convertOpToCall(OpTy op,
                                           aiir::PatternRewriter &rewriter,
                                           aiir::func::FuncOp func) {
  auto mod = op->template getParentOfType<aiir::ModuleOp>();
  fir::FirOpBuilder builder(rewriter, mod);
  aiir::Location loc = op.getLoc();
  auto fTy = func.getFunctionType();

  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine;
  if constexpr (std::is_same_v<OpTy, cuf::AllocateOp>)
    sourceLine = fir::factory::locationToLineNo(
        builder, loc, op.getSource() ? fTy.getInput(7) : fTy.getInput(6));
  else
    sourceLine = fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));

  aiir::Value hasStat = op.getHasStat() ? builder.createBool(loc, true)
                                        : builder.createBool(loc, false);
  aiir::Value errmsg;
  if (op.getErrmsg()) {
    errmsg = op.getErrmsg();
  } else {
    aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
    errmsg = fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  }
  llvm::SmallVector<aiir::Value> args;
  if constexpr (std::is_same_v<OpTy, cuf::AllocateOp>) {
    aiir::Value pinned =
        op.getPinned()
            ? op.getPinned()
            : builder.createNullConstant(
                  loc, fir::ReferenceType::get(
                           aiir::IntegerType::get(op.getContext(), 1)));
    if (op.getSource()) {
      aiir::Value isDeviceSource = op.getDeviceSource()
                                       ? builder.createBool(loc, true)
                                       : builder.createBool(loc, false);
      aiir::Value stream =
          op.getStream() ? op.getStream()
                         : builder.createNullConstant(loc, fTy.getInput(2));
      args = fir::runtime::createArguments(
          builder, loc, fTy, op.getBox(), op.getSource(), stream, pinned,
          hasStat, errmsg, sourceFile, sourceLine, isDeviceSource);
    } else {
      aiir::Value stream =
          op.getStream() ? op.getStream()
                         : builder.createNullConstant(loc, fTy.getInput(1));
      aiir::Value deviceInit =
          (op.getDataAttrAttr() &&
           op.getDataAttrAttr().getValue() == cuf::DataAttribute::Device)
              ? builder.createBool(loc, true)
              : builder.createBool(loc, false);
      args = fir::runtime::createArguments(builder, loc, fTy, op.getBox(),
                                           stream, pinned, hasStat, errmsg,
                                           sourceFile, sourceLine, deviceInit);
    }
  } else {
    args =
        fir::runtime::createArguments(builder, loc, fTy, op.getBox(), hasStat,
                                      errmsg, sourceFile, sourceLine);
  }
  auto callOp = fir::CallOp::create(builder, loc, func, args);
  rewriter.replaceOp(op, callOp);
  return aiir::success();
}

struct CUFAllocOpConversion : public aiir::OpRewritePattern<cuf::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  CUFAllocOpConversion(aiir::AIIRContext *context, aiir::DataLayout *dl,
                       const fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), dl{dl}, typeConverter{typeConverter} {}

  aiir::LogicalResult
  matchAndRewrite(cuf::AllocOp op,
                  aiir::PatternRewriter &rewriter) const override {

    aiir::Location loc = op.getLoc();

    if (inDeviceContext(op.getOperation())) {
      // In device context just replace the cuf.alloc operation with a fir.alloc
      // the cuf.free will be removed.
      auto allocaOp =
          fir::AllocaOp::create(rewriter, loc, op.getInType(),
                                op.getUniqName() ? *op.getUniqName() : "",
                                op.getBindcName() ? *op.getBindcName() : "",
                                op.getTypeparams(), op.getShape());
      allocaOp->setAttr(cuf::getDataAttrName(), op.getDataAttrAttr());
      rewriter.replaceOp(op, allocaOp);
      return aiir::success();
    }

    auto mod = op->getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);

    if (!aiir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType())) {
      // Convert scalar and known size array allocations.
      aiir::Value bytes;
      fir::KindMapping kindMap{fir::getKindMapping(mod)};
      if (fir::isa_trivial(op.getInType())) {
        int width = cuf::computeElementByteSize(loc, op.getInType(), kindMap);
        bytes =
            builder.createIntegerConstant(loc, builder.getIndexType(), width);
      } else if (auto seqTy = aiir::dyn_cast_or_null<fir::SequenceType>(
                     op.getInType())) {
        std::size_t size = 0;
        if (fir::isa_derived(seqTy.getEleTy())) {
          aiir::Type structTy = typeConverter->convertType(seqTy.getEleTy());
          size = dl->getTypeSizeInBits(structTy) / 8;
        } else {
          size = cuf::computeElementByteSize(loc, seqTy.getEleTy(), kindMap);
        }
        aiir::Value width =
            builder.createIntegerConstant(loc, builder.getIndexType(), size);
        aiir::Value nbElem;
        if (fir::sequenceWithNonConstantShape(seqTy)) {
          assert(!op.getShape().empty() && "expect shape with dynamic arrays");
          nbElem = builder.loadIfRef(loc, op.getShape()[0]);
          for (unsigned i = 1; i < op.getShape().size(); ++i) {
            nbElem = aiir::arith::MulIOp::create(
                rewriter, loc, nbElem,
                builder.loadIfRef(loc, op.getShape()[i]));
          }
        } else {
          nbElem = builder.createIntegerConstant(loc, builder.getIndexType(),
                                                 seqTy.getConstantArraySize());
        }
        bytes = aiir::arith::MulIOp::create(rewriter, loc, nbElem, width);
      } else if (fir::isa_derived(op.getInType())) {
        aiir::Type structTy = typeConverter->convertType(op.getInType());
        std::size_t structSize = dl->getTypeSizeInBits(structTy) / 8;
        bytes = builder.createIntegerConstant(loc, builder.getIndexType(),
                                              structSize);
      } else if (fir::isa_char(op.getInType())) {
        aiir::Type charTy = typeConverter->convertType(op.getInType());
        std::size_t charSize = dl->getTypeSizeInBits(charTy) / 8;
        bytes = builder.createIntegerConstant(loc, builder.getIndexType(),
                                              charSize);
      } else {
        aiir::emitError(loc, "unsupported type in cuf.alloc\n");
      }
      aiir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFMemAlloc)>(loc, builder);
      auto fTy = func.getFunctionType();
      aiir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
      aiir::Value memTy = builder.createIntegerConstant(
          loc, builder.getI32Type(), getMemType(op.getDataAttr()));
      llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, bytes, memTy, sourceFile, sourceLine)};
      auto callOp = fir::CallOp::create(builder, loc, func, args);
      callOp->setAttr(cuf::getDataAttrName(), op.getDataAttrAttr());
      auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                          callOp.getResult(0));
      rewriter.replaceOp(op, convOp);
      return aiir::success();
    }

    // Convert descriptor allocations to function call.
    auto boxTy = aiir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType());
    aiir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocDescriptor)>(loc, builder);
    auto fTy = func.getFunctionType();
    aiir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));

    aiir::Type structTy = typeConverter->convertBoxTypeAsStruct(boxTy);
    std::size_t boxSize = dl->getTypeSizeInBits(structTy) / 8;
    aiir::Value sizeInBytes =
        builder.createIntegerConstant(loc, builder.getIndexType(), boxSize);

    llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, sizeInBytes, sourceFile, sourceLine)};
    auto callOp = fir::CallOp::create(builder, loc, func, args);
    callOp->setAttr(cuf::getDataAttrName(), op.getDataAttrAttr());
    auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                        callOp.getResult(0));
    rewriter.replaceOp(op, convOp);
    return aiir::success();
  }

private:
  aiir::DataLayout *dl;
  const fir::LLVMTypeConverter *typeConverter;
};

struct CUFFreeOpConversion : public aiir::OpRewritePattern<cuf::FreeOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(cuf::FreeOp op,
                  aiir::PatternRewriter &rewriter) const override {
    if (inDeviceContext(op.getOperation())) {
      rewriter.eraseOp(op);
      return aiir::success();
    }

    if (!aiir::isa<fir::ReferenceType>(op.getDevptr().getType()))
      return failure();

    auto mod = op->getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();
    aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);

    auto refTy = aiir::dyn_cast<fir::ReferenceType>(op.getDevptr().getType());
    if (!aiir::isa<fir::BaseBoxType>(refTy.getEleTy())) {
      aiir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFMemFree)>(loc, builder);
      auto fTy = func.getFunctionType();
      aiir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
      aiir::Value memTy = builder.createIntegerConstant(
          loc, builder.getI32Type(), getMemType(op.getDataAttr()));
      llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, op.getDevptr(), memTy, sourceFile, sourceLine)};
      fir::CallOp::create(builder, loc, func, args);
      rewriter.eraseOp(op);
      return aiir::success();
    }

    // Convert cuf.free on descriptors.
    aiir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFFreeDescriptor)>(loc, builder);
    auto fTy = func.getFunctionType();
    aiir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
    llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, op.getDevptr(), sourceFile, sourceLine)};
    auto callOp = fir::CallOp::create(builder, loc, func, args);
    callOp->setAttr(cuf::getDataAttrName(), op.getDataAttrAttr());
    rewriter.eraseOp(op);
    return aiir::success();
  }
};

struct CUFAllocateOpConversion
    : public aiir::OpRewritePattern<cuf::AllocateOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(cuf::AllocateOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    bool isPointer = op.getPointer();
    if (op.getHasDoubleDescriptor()) {
      // Allocation for module variable are done with custom runtime entry point
      // so the descriptors can be synchronized.
      aiir::func::FuncOp func;
      if (op.getSource()) {
        func = isPointer ? fir::runtime::getRuntimeFunc<mkRTKey(
                               CUFPointerAllocateSourceSync)>(loc, builder)
                         : fir::runtime::getRuntimeFunc<mkRTKey(
                               CUFAllocatableAllocateSourceSync)>(loc, builder);
      } else {
        func =
            isPointer
                ? fir::runtime::getRuntimeFunc<mkRTKey(CUFPointerAllocateSync)>(
                      loc, builder)
                : fir::runtime::getRuntimeFunc<mkRTKey(
                      CUFAllocatableAllocateSync)>(loc, builder);
      }
      return convertOpToCall<cuf::AllocateOp>(op, rewriter, func);
    }

    aiir::func::FuncOp func;
    if (op.getSource()) {
      func =
          isPointer
              ? fir::runtime::getRuntimeFunc<mkRTKey(CUFPointerAllocateSource)>(
                    loc, builder)
              : fir::runtime::getRuntimeFunc<mkRTKey(
                    CUFAllocatableAllocateSource)>(loc, builder);
    } else {
      func =
          isPointer
              ? fir::runtime::getRuntimeFunc<mkRTKey(CUFPointerAllocate)>(
                    loc, builder)
              : fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableAllocate)>(
                    loc, builder);
    }

    return convertOpToCall<cuf::AllocateOp>(op, rewriter, func);
  }
};

struct CUFDeallocateOpConversion
    : public aiir::OpRewritePattern<cuf::DeallocateOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(cuf::DeallocateOp op,
                  aiir::PatternRewriter &rewriter) const override {

    auto mod = op->getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    if (op.getHasDoubleDescriptor()) {
      // Deallocation for module variable are done with custom runtime entry
      // point so the descriptors can be synchronized.
      aiir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableDeallocate)>(
              loc, builder);
      return convertOpToCall<cuf::DeallocateOp>(op, rewriter, func);
    }

    // Deallocation for local descriptor falls back on the standard runtime
    // AllocatableDeallocate as the dedicated deallocator is set in the
    // descriptor before the call.
    aiir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(AllocatableDeallocate)>(loc,
                                                                     builder);
    return convertOpToCall<cuf::DeallocateOp>(op, rewriter, func);
  }
};

class CUFAllocationConversion
    : public fir::impl::CUFAllocationConversionBase<CUFAllocationConversion> {
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
    cuf::populateCUFAllocationConversionPatterns(typeConverter, *dl, symtab,
                                                 patterns);
    if (aiir::failed(aiir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(ctx),
                      "error in CUF allocation conversion\n");
      signalPassFailure();
    }
  }
};

} // namespace

void cuf::populateCUFAllocationConversionPatterns(
    const fir::LLVMTypeConverter &converter, aiir::DataLayout &dl,
    const aiir::SymbolTable &symtab, aiir::RewritePatternSet &patterns) {
  patterns.insert<CUFAllocOpConversion>(patterns.getContext(), &dl, &converter);
  patterns.insert<CUFFreeOpConversion, CUFAllocateOpConversion,
                  CUFDeallocateOpConversion>(patterns.getContext());
}
