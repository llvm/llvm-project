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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CUFALLOCATIONCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;
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

static bool inDeviceContext(mlir::Operation *op) {
  if (op->getParentOfType<cuf::KernelOp>())
    return true;
  if (auto funcOp = op->getParentOfType<mlir::gpu::GPUFuncOp>())
    return true;
  if (auto funcOp = op->getParentOfType<mlir::gpu::LaunchOp>())
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

template <typename OpTy>
static mlir::LogicalResult convertOpToCall(OpTy op,
                                           mlir::PatternRewriter &rewriter,
                                           mlir::func::FuncOp func) {
  auto mod = op->template getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder builder(rewriter, mod);
  mlir::Location loc = op.getLoc();
  auto fTy = func.getFunctionType();

  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine;
  if constexpr (std::is_same_v<OpTy, cuf::AllocateOp>)
    sourceLine = fir::factory::locationToLineNo(
        builder, loc, op.getSource() ? fTy.getInput(7) : fTy.getInput(6));
  else
    sourceLine = fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));

  mlir::Value hasStat = op.getHasStat() ? builder.createBool(loc, true)
                                        : builder.createBool(loc, false);
  mlir::Value errmsg;
  if (op.getErrmsg()) {
    errmsg = op.getErrmsg();
  } else {
    mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
    errmsg = fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  }
  llvm::SmallVector<mlir::Value> args;
  if constexpr (std::is_same_v<OpTy, cuf::AllocateOp>) {
    mlir::Value pinned =
        op.getPinned()
            ? op.getPinned()
            : builder.createNullConstant(
                  loc, fir::ReferenceType::get(
                           mlir::IntegerType::get(op.getContext(), 1)));
    if (op.getSource()) {
      mlir::Value isDeviceSource = op.getDeviceSource()
                                       ? builder.createBool(loc, true)
                                       : builder.createBool(loc, false);
      mlir::Value stream =
          op.getStream() ? op.getStream()
                         : builder.createNullConstant(loc, fTy.getInput(2));
      args = fir::runtime::createArguments(
          builder, loc, fTy, op.getBox(), op.getSource(), stream, pinned,
          hasStat, errmsg, sourceFile, sourceLine, isDeviceSource);
    } else {
      mlir::Value stream =
          op.getStream() ? op.getStream()
                         : builder.createNullConstant(loc, fTy.getInput(1));
      mlir::Value deviceInit =
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
  return mlir::success();
}

struct CUFAllocOpConversion : public mlir::OpRewritePattern<cuf::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  CUFAllocOpConversion(mlir::MLIRContext *context, mlir::DataLayout *dl,
                       const fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), dl{dl}, typeConverter{typeConverter} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();

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
      return mlir::success();
    }

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);

    if (!mlir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType())) {
      // Convert scalar and known size array allocations.
      mlir::Value bytes;
      fir::KindMapping kindMap{fir::getKindMapping(mod)};
      if (fir::isa_trivial(op.getInType())) {
        int width = cuf::computeElementByteSize(loc, op.getInType(), kindMap);
        bytes =
            builder.createIntegerConstant(loc, builder.getIndexType(), width);
      } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
                     op.getInType())) {
        std::size_t size = 0;
        if (fir::isa_derived(seqTy.getEleTy())) {
          mlir::Type structTy = typeConverter->convertType(seqTy.getEleTy());
          size = dl->getTypeSizeInBits(structTy) / 8;
        } else {
          size = cuf::computeElementByteSize(loc, seqTy.getEleTy(), kindMap);
        }
        mlir::Value width =
            builder.createIntegerConstant(loc, builder.getIndexType(), size);
        mlir::Value nbElem;
        if (fir::sequenceWithNonConstantShape(seqTy)) {
          assert(!op.getShape().empty() && "expect shape with dynamic arrays");
          nbElem = builder.loadIfRef(loc, op.getShape()[0]);
          for (unsigned i = 1; i < op.getShape().size(); ++i) {
            nbElem = mlir::arith::MulIOp::create(
                rewriter, loc, nbElem,
                builder.loadIfRef(loc, op.getShape()[i]));
          }
        } else {
          nbElem = builder.createIntegerConstant(loc, builder.getIndexType(),
                                                 seqTy.getConstantArraySize());
        }
        bytes = mlir::arith::MulIOp::create(rewriter, loc, nbElem, width);
      } else if (fir::isa_derived(op.getInType())) {
        mlir::Type structTy = typeConverter->convertType(op.getInType());
        std::size_t structSize = dl->getTypeSizeInBits(structTy) / 8;
        bytes = builder.createIntegerConstant(loc, builder.getIndexType(),
                                              structSize);
      } else if (fir::isa_char(op.getInType())) {
        mlir::Type charTy = typeConverter->convertType(op.getInType());
        std::size_t charSize = dl->getTypeSizeInBits(charTy) / 8;
        bytes = builder.createIntegerConstant(loc, builder.getIndexType(),
                                              charSize);
      } else {
        mlir::emitError(loc, "unsupported type in cuf.alloc\n");
      }
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFMemAlloc)>(loc, builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
      mlir::Value memTy = builder.createIntegerConstant(
          loc, builder.getI32Type(), getMemType(op.getDataAttr()));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, bytes, memTy, sourceFile, sourceLine)};
      auto callOp = fir::CallOp::create(builder, loc, func, args);
      callOp->setAttr(cuf::getDataAttrName(), op.getDataAttrAttr());
      auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                          callOp.getResult(0));
      rewriter.replaceOp(op, convOp);
      return mlir::success();
    }

    // Convert descriptor allocations to function call.
    auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType());
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocDescriptor)>(loc, builder);
    auto fTy = func.getFunctionType();
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));

    mlir::Type structTy = typeConverter->convertBoxTypeAsStruct(boxTy);
    std::size_t boxSize = dl->getTypeSizeInBits(structTy) / 8;
    mlir::Value sizeInBytes =
        builder.createIntegerConstant(loc, builder.getIndexType(), boxSize);

    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, sizeInBytes, sourceFile, sourceLine)};
    auto callOp = fir::CallOp::create(builder, loc, func, args);
    callOp->setAttr(cuf::getDataAttrName(), op.getDataAttrAttr());
    auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                        callOp.getResult(0));
    rewriter.replaceOp(op, convOp);
    return mlir::success();
  }

private:
  mlir::DataLayout *dl;
  const fir::LLVMTypeConverter *typeConverter;
};

struct CUFFreeOpConversion : public mlir::OpRewritePattern<cuf::FreeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::FreeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (inDeviceContext(op.getOperation())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (!mlir::isa<fir::ReferenceType>(op.getDevptr().getType()))
      return failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);

    auto refTy = mlir::dyn_cast<fir::ReferenceType>(op.getDevptr().getType());
    if (!mlir::isa<fir::BaseBoxType>(refTy.getEleTy())) {
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFMemFree)>(loc, builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
      mlir::Value memTy = builder.createIntegerConstant(
          loc, builder.getI32Type(), getMemType(op.getDataAttr()));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, op.getDevptr(), memTy, sourceFile, sourceLine)};
      fir::CallOp::create(builder, loc, func, args);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Convert cuf.free on descriptors.
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFFreeDescriptor)>(loc, builder);
    auto fTy = func.getFunctionType();
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, op.getDevptr(), sourceFile, sourceLine)};
    auto callOp = fir::CallOp::create(builder, loc, func, args);
    callOp->setAttr(cuf::getDataAttrName(), op.getDataAttrAttr());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CUFAllocateOpConversion
    : public mlir::OpRewritePattern<cuf::AllocateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::AllocateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    bool isPointer = op.getPointer();
    if (op.getHasDoubleDescriptor()) {
      // Allocation for module variable are done with custom runtime entry point
      // so the descriptors can be synchronized.
      mlir::func::FuncOp func;
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

    mlir::func::FuncOp func;
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
    : public mlir::OpRewritePattern<cuf::DeallocateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::DeallocateOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    if (op.getHasDoubleDescriptor()) {
      // Deallocation for module variable are done with custom runtime entry
      // point so the descriptors can be synchronized.
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableDeallocate)>(
              loc, builder);
      return convertOpToCall<cuf::DeallocateOp>(op, rewriter, func);
    }

    // Deallocation for local descriptor falls back on the standard runtime
    // AllocatableDeallocate as the dedicated deallocator is set in the
    // descriptor before the call.
    mlir::func::FuncOp func =
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
    cuf::populateCUFAllocationConversionPatterns(typeConverter, *dl, symtab,
                                                 patterns);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF allocation conversion\n");
      signalPassFailure();
    }
  }
};

} // namespace

void cuf::populateCUFAllocationConversionPatterns(
    const fir::LLVMTypeConverter &converter, mlir::DataLayout &dl,
    const mlir::SymbolTable &symtab, mlir::RewritePatternSet &patterns) {
  patterns.insert<CUFAllocOpConversion>(patterns.getContext(), &dl, &converter);
  patterns.insert<CUFFreeOpConversion, CUFAllocateOpConversion,
                  CUFDeallocateOpConversion>(patterns.getContext());
}
