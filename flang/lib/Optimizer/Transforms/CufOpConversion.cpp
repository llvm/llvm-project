//===-- CufOpConversion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/CufOpConversion.h"
#include "flang/Common/Fortran.h"
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
#include "flang/Runtime/allocatable.h"
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

static inline unsigned getMemType(cuf::DataAttribute attr) {
  if (attr == cuf::DataAttribute::Device)
    return kMemTypeDevice;
  if (attr == cuf::DataAttribute::Managed)
    return kMemTypeManaged;
  if (attr == cuf::DataAttribute::Unified)
    return kMemTypeUnified;
  if (attr == cuf::DataAttribute::Pinned)
    return kMemTypePinned;
  llvm::report_fatal_error("unsupported memory type");
}

template <typename OpTy>
static bool isPinned(OpTy op) {
  if (op.getDataAttr() && *op.getDataAttr() == cuf::DataAttribute::Pinned)
    return true;
  return false;
}

template <typename OpTy>
static bool hasDoubleDescriptors(OpTy op) {
  if (auto declareOp =
          mlir::dyn_cast_or_null<fir::DeclareOp>(op.getBox().getDefiningOp())) {
    if (mlir::isa_and_nonnull<fir::AddrOfOp>(
            declareOp.getMemref().getDefiningOp())) {
      if (isPinned(declareOp))
        return false;
      return true;
    }
  } else if (auto declareOp = mlir::dyn_cast_or_null<hlfir::DeclareOp>(
                 op.getBox().getDefiningOp())) {
    if (mlir::isa_and_nonnull<fir::AddrOfOp>(
            declareOp.getMemref().getDefiningOp())) {
      if (isPinned(declareOp))
        return false;
      return true;
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
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));

  mlir::Value hasStat = op.getHasStat() ? builder.createBool(loc, true)
                                        : builder.createBool(loc, false);

  mlir::Value errmsg;
  if (op.getErrmsg()) {
    errmsg = op.getErrmsg();
  } else {
    mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
    errmsg = builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  }
  llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, op.getBox(), hasStat, errmsg, sourceFile, sourceLine)};
  auto callOp = builder.create<fir::CallOp>(loc, func, args);
  rewriter.replaceOp(op, callOp);
  return mlir::success();
}

struct CufAllocateOpConversion
    : public mlir::OpRewritePattern<cuf::AllocateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::AllocateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // TODO: Allocation with source will need a new entry point in the runtime.
    if (op.getSource())
      return mlir::failure();

    // TODO: Allocation using different stream.
    if (op.getStream())
      return mlir::failure();

    // TODO: Pinned is a reference to a logical value that can be set to true
    // when pinned allocation succeed. This will require a new entry point.
    if (op.getPinned())
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    if (hasDoubleDescriptors(op)) {
      // Allocation for module variable are done with custom runtime entry point
      // so the descriptors can be synchronized.
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableAllocate)>(
              loc, builder);
      return convertOpToCall(op, rewriter, func);
    }

    // Allocation for local descriptor falls back on the standard runtime
    // AllocatableAllocate as the dedicated allocator is set in the descriptor
    // before the call.
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(AllocatableAllocate)>(loc,
                                                                   builder);
    return convertOpToCall<cuf::AllocateOp>(op, rewriter, func);
  }
};

struct CufDeallocateOpConversion
    : public mlir::OpRewritePattern<cuf::DeallocateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::DeallocateOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    if (hasDoubleDescriptors(op)) {
      // Deallocation for module variable are done with custom runtime entry
      // point so the descriptors can be synchronized.
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableDeallocate)>(
              loc, builder);
      return convertOpToCall(op, rewriter, func);
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

static bool inDeviceContext(mlir::Operation *op) {
  if (op->getParentOfType<cuf::KernelOp>())
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

static int computeWidth(mlir::Location loc, mlir::Type type,
                        fir::KindMapping &kindMap) {
  auto eleTy = fir::unwrapSequenceType(type);
  int width = 0;
  if (auto t{mlir::dyn_cast<mlir::IntegerType>(eleTy)}) {
    width = t.getWidth() / 8;
  } else if (auto t{mlir::dyn_cast<mlir::FloatType>(eleTy)}) {
    width = t.getWidth() / 8;
  } else if (eleTy.isInteger(1)) {
    width = 1;
  } else if (auto t{mlir::dyn_cast<fir::LogicalType>(eleTy)}) {
    int kind = t.getFKind();
    width = kindMap.getLogicalBitsize(kind) / 8;
  } else if (auto t{mlir::dyn_cast<mlir::ComplexType>(eleTy)}) {
    int elemSize =
        mlir::cast<mlir::FloatType>(t.getElementType()).getWidth() / 8;
    width = 2 * elemSize;
  } else {
    llvm::report_fatal_error("unsupported type");
  }
  return width;
}

struct CufAllocOpConversion : public mlir::OpRewritePattern<cuf::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  CufAllocOpConversion(mlir::MLIRContext *context, mlir::DataLayout *dl,
                       fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), dl{dl}, typeConverter{typeConverter} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (inDeviceContext(op.getOperation())) {
      // In device context just replace the cuf.alloc operation with a fir.alloc
      // the cuf.free will be removed.
      rewriter.replaceOpWithNewOp<fir::AllocaOp>(
          op, op.getInType(), op.getUniqName() ? *op.getUniqName() : "",
          op.getBindcName() ? *op.getBindcName() : "", op.getTypeparams(),
          op.getShape());
      return mlir::success();
    }

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);

    if (!mlir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType())) {
      // Convert scalar and known size array allocations.
      mlir::Value bytes;
      fir::KindMapping kindMap{fir::getKindMapping(mod)};
      if (fir::isa_trivial(op.getInType())) {
        int width = computeWidth(loc, op.getInType(), kindMap);
        bytes =
            builder.createIntegerConstant(loc, builder.getIndexType(), width);
      } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
                     op.getInType())) {
        mlir::Value width = builder.createIntegerConstant(
            loc, builder.getIndexType(),
            computeWidth(loc, seqTy.getEleTy(), kindMap));
        mlir::Value nbElem;
        if (fir::sequenceWithNonConstantShape(seqTy)) {
          assert(!op.getShape().empty() && "expect shape with dynamic arrays");
          nbElem = builder.loadIfRef(loc, op.getShape()[0]);
          for (unsigned i = 1; i < op.getShape().size(); ++i) {
            nbElem = rewriter.create<mlir::arith::MulIOp>(
                loc, nbElem, builder.loadIfRef(loc, op.getShape()[i]));
          }
        } else {
          nbElem = builder.createIntegerConstant(loc, builder.getIndexType(),
                                                 seqTy.getConstantArraySize());
        }
        bytes = rewriter.create<mlir::arith::MulIOp>(loc, nbElem, width);
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
      auto callOp = builder.create<fir::CallOp>(loc, func, args);
      auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                          callOp.getResult(0));
      rewriter.replaceOp(op, convOp);
      return mlir::success();
    }

    // Convert descriptor allocations to function call.
    auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType());
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocDesciptor)>(loc, builder);
    auto fTy = func.getFunctionType();
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));

    mlir::Type structTy = typeConverter->convertBoxTypeAsStruct(boxTy);
    std::size_t boxSize = dl->getTypeSizeInBits(structTy) / 8;
    mlir::Value sizeInBytes =
        builder.createIntegerConstant(loc, builder.getIndexType(), boxSize);

    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, sizeInBytes, sourceFile, sourceLine)};
    auto callOp = builder.create<fir::CallOp>(loc, func, args);
    auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                        callOp.getResult(0));
    rewriter.replaceOp(op, convOp);
    return mlir::success();
  }

private:
  mlir::DataLayout *dl;
  fir::LLVMTypeConverter *typeConverter;
};

struct CufFreeOpConversion : public mlir::OpRewritePattern<cuf::FreeOp> {
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
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Convert cuf.free on descriptors.
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFFreeDesciptor)>(loc, builder);
    auto fTy = func.getFunctionType();
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, op.getDevptr(), sourceFile, sourceLine)};
    builder.create<fir::CallOp>(loc, func, args);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static mlir::Value createConvertOp(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Type toTy,
                                   mlir::Value val) {
  if (val.getType() != toTy)
    return rewriter.create<fir::ConvertOp>(loc, toTy, val);
  return val;
}

struct CufDataTransferOpConversion
    : public mlir::OpRewritePattern<cuf::DataTransferOp> {
  using OpRewritePattern::OpRewritePattern;

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
        // TODO: scalar to array data transfer.
        mlir::emitError(loc,
                        "not yet implemented: scalar to array data transfer\n");
        return mlir::failure();
      }

      mlir::Type i64Ty = builder.getI64Type();
      mlir::Value nbElement;
      if (op.getShape()) {
        auto shapeOp =
            mlir::dyn_cast<fir::ShapeOp>(op.getShape().getDefiningOp());
        nbElement = rewriter.create<fir::ConvertOp>(loc, i64Ty,
                                                    shapeOp.getExtents()[0]);
        for (unsigned i = 1; i < shapeOp.getExtents().size(); ++i) {
          auto operand = rewriter.create<fir::ConvertOp>(
              loc, i64Ty, shapeOp.getExtents()[i]);
          nbElement =
              rewriter.create<mlir::arith::MulIOp>(loc, nbElement, operand);
        }
      } else {
        if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(dstTy))
          nbElement = builder.createIntegerConstant(
              loc, i64Ty, seqTy.getConstantArraySize());
      }
      int width = computeWidth(loc, dstTy, kindMap);
      mlir::Value widthValue = rewriter.create<mlir::arith::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, width));
      mlir::Value bytes =
          nbElement
              ? rewriter.create<mlir::arith::MulIOp>(loc, nbElement, widthValue)
              : widthValue;

      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferPtrPtr)>(loc,
                                                                       builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));

      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, op.getDst(), op.getSrc(), bytes, modeValue,
          sourceFile, sourceLine)};
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Conversion of data transfer involving at least one descriptor.
    if (mlir::isa<fir::BaseBoxType>(srcTy) &&
        mlir::isa<fir::BaseBoxType>(dstTy)) {
      // Transfer between two descriptor.
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferDescDesc)>(
              loc, builder);

      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
      mlir::Value dst = builder.loadIfRef(loc, op.getDst());
      mlir::Value src = builder.loadIfRef(loc, op.getSrc());
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
    } else if (mlir::isa<fir::BaseBoxType>(dstTy) && fir::isa_trivial(srcTy)) {
      // Scalar to descriptor transfer.
      mlir::Value val = op.getSrc();
      if (op.getSrc().getDefiningOp() &&
          mlir::isa<mlir::arith::ConstantOp>(op.getSrc().getDefiningOp())) {
        mlir::Value alloc = builder.createTemporary(loc, srcTy);
        builder.create<fir::StoreOp>(loc, op.getSrc(), alloc);
        val = alloc;
      }

      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFMemsetDescriptor)>(loc,
                                                                     builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
      mlir::Value dst = builder.loadIfRef(loc, op.getDst());
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, dst, val, sourceFile, sourceLine)};
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
    } else {
      // Type used to compute the width.
      mlir::Type computeType = dstTy;
      auto seqTy = mlir::dyn_cast<fir::SequenceType>(dstTy);
      bool dstIsDesc = false;
      if (mlir::isa<fir::BaseBoxType>(dstTy)) {
        dstIsDesc = true;
        computeType = srcTy;
        seqTy = mlir::dyn_cast<fir::SequenceType>(srcTy);
      }
      int width = computeWidth(loc, computeType, kindMap);

      mlir::Value nbElement;
      mlir::Type idxTy = rewriter.getIndexType();
      if (!op.getShape()) {
        nbElement = rewriter.create<mlir::arith::ConstantOp>(
            loc, idxTy,
            rewriter.getIntegerAttr(idxTy, seqTy.getConstantArraySize()));
      } else {
        auto shapeOp =
            mlir::dyn_cast<fir::ShapeOp>(op.getShape().getDefiningOp());
        nbElement =
            createConvertOp(rewriter, loc, idxTy, shapeOp.getExtents()[0]);
        for (unsigned i = 1; i < shapeOp.getExtents().size(); ++i) {
          auto operand =
              createConvertOp(rewriter, loc, idxTy, shapeOp.getExtents()[i]);
          nbElement =
              rewriter.create<mlir::arith::MulIOp>(loc, nbElement, operand);
        }
      }

      mlir::Value widthValue = rewriter.create<mlir::arith::ConstantOp>(
          loc, idxTy, rewriter.getIntegerAttr(idxTy, width));
      mlir::Value bytes =
          rewriter.create<mlir::arith::MulIOp>(loc, nbElement, widthValue);

      mlir::func::FuncOp func =
          dstIsDesc
              ? fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferDescPtr)>(
                    loc, builder)
              : fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferPtrDesc)>(
                    loc, builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
      mlir::Value dst =
          dstIsDesc ? builder.loadIfRef(loc, op.getDst()) : op.getDst();
      mlir::Value src = mlir::isa<fir::BaseBoxType>(srcTy)
                            ? builder.loadIfRef(loc, op.getSrc())
                            : op.getSrc();
      llvm::SmallVector<mlir::Value> args{
          fir::runtime::createArguments(builder, loc, fTy, dst, src, bytes,
                                        modeValue, sourceFile, sourceLine)};
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
    }
    return mlir::success();
  }
};

class CufOpConversion : public fir::impl::CufOpConversionBase<CufOpConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);

    mlir::Operation *op = getOperation();
    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();

    std::optional<mlir::DataLayout> dl =
        fir::support::getOrSetDataLayout(module, /*allowDefaultLayout=*/false);
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                         /*forceUnifiedTBAATree=*/false, *dl);
    target.addLegalDialect<fir::FIROpsDialect, mlir::arith::ArithDialect>();
    cuf::populateCUFToFIRConversionPatterns(typeConverter, *dl, patterns);
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
    fir::LLVMTypeConverter &converter, mlir::DataLayout &dl,
    mlir::RewritePatternSet &patterns) {
  patterns.insert<CufAllocOpConversion>(patterns.getContext(), &dl, &converter);
  patterns.insert<CufAllocateOpConversion, CufDeallocateOpConversion,
                  CufFreeOpConversion, CufDataTransferOpConversion>(
      patterns.getContext());
}
