//===-- FIRToCoreMLIR.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_FIRTOCOREMLIRPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class FIRToCoreMLIRPass
    : public fir::impl::FIRToCoreMLIRPassBase<FIRToCoreMLIRPass> {
public:
  void runOnOperation() override;
};

class FIRLoadOpLowering : public mlir::OpConversionPattern<fir::LoadOp> {
public:
  using mlir::OpConversionPattern<fir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!getTypeConverter()->convertType(op.getMemref()))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getMemref(),
                                                      mlir::ValueRange{});
    return mlir::success();
  }
};

class FIRStoreOpLowering : public mlir::OpConversionPattern<fir::StoreOp> {
public:
  using mlir::OpConversionPattern<fir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!getTypeConverter()->convertType(op.getMemref()))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), mlir::ValueRange{});
    return mlir::success();
  }
};

class FIRConvertOpLowering : public mlir::OpConversionPattern<fir::ConvertOp> {
public:
  using mlir::OpConversionPattern<fir::ConvertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::ConvertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcType = getTypeConverter()->convertType(op.getValue().getType());
    auto dstType = getTypeConverter()->convertType(op.getType());

    if (!srcType || !dstType)
      return mlir::failure();

    auto srcVal = adaptor.getValue();

    if (srcType == dstType) {
      rewriter.replaceOp(op, mlir::ValueRange{srcVal});
    } else if (srcType.isIntOrIndex() && dstType.isIntOrIndex()) {
      if (srcType.isIndex() || dstType.isIndex()) {
        rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(op, dstType,
                                                              srcVal);
      } else if (srcType.getIntOrFloatBitWidth() <
                 dstType.getIntOrFloatBitWidth()) {
        rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(op, dstType, srcVal);
      } else {
        rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(op, dstType, srcVal);
      }
    } else if (srcType.isFloat() && dstType.isFloat()) {
      if (srcType.getIntOrFloatBitWidth() < dstType.getIntOrFloatBitWidth()) {
        rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(op, dstType, srcVal);
      } else {
        rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(op, dstType, srcVal);
      }
    } else {
      return mlir::failure();
    }

    return mlir::success();
  }
};

class FIRAllocOpLowering : public mlir::OpConversionPattern<fir::AllocaOp> {
public:
  using mlir::OpConversionPattern<fir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!mlir::MemRefType::isValidElementType(op.getAllocatedType()) ||
        op.hasLenParams())
      return mlir::failure();

    auto dstType = getTypeConverter()->convertType(op.getType());
    auto allocaOp = mlir::memref::AllocaOp::create(
        rewriter, op.getLoc(),
        mlir::MemRefType::get({}, op.getAllocatedType()));
    allocaOp->setAttrs(op->getAttrs());
    rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, dstType, allocaOp);
    return mlir::success();
  }
};

class FIRXArrayCoorOpLowering
    : public mlir::OpConversionPattern<fir::cg::XArrayCoorOp> {
public:
  using mlir::OpConversionPattern<fir::cg::XArrayCoorOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::cg::XArrayCoorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!getTypeConverter()->convertType(op.getMemref()))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    auto metadata = mlir::memref::ExtractStridedMetadataOp::create(
        rewriter, loc, adaptor.getMemref());
    auto base = metadata.getBaseBuffer();
    auto offset = metadata.getOffset();
    mlir::ValueRange shape = adaptor.getShape();
    unsigned rank = op.getRank();

    assert(rank > 0 && "expected rank to be greater than zero");

    auto sizes = llvm::to_vector_of<mlir::OpFoldResult>(llvm::reverse(shape));
    mlir::SmallVector<mlir::OpFoldResult> strides(rank);

    strides[rank - 1] = rewriter.getIndexAttr(1);
    mlir::Value stride = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    for (unsigned i = 1; i < rank; ++i) {
      stride = mlir::arith::MulIOp::create(rewriter, loc, stride, shape[i - 1]);
      strides[rank - 1 - i] = stride;
    }

    mlir::Value memref = mlir::memref::ReinterpretCastOp::create(
        rewriter, loc, base, offset, sizes, strides);

    mlir::SmallVector<mlir::OpFoldResult> oneAttrs(rank,
                                                   rewriter.getIndexAttr(1));
    auto one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto offsets = llvm::map_to_vector(
        llvm::reverse(adaptor.getIndices()),
        [&](mlir::Value idx) -> mlir::OpFoldResult {
          if (idx.getType().isInteger())
            idx = mlir::arith::IndexCastOp::create(
                rewriter, loc, rewriter.getIndexType(), idx);

          assert(idx.getType().isIndex() && "expected index type");
          idx = mlir::arith::SubIOp::create(rewriter, loc, idx, one);
          return idx;
        });

    auto subview = mlir::memref::SubViewOp::create(
        rewriter, loc,
        mlir::cast<mlir::MemRefType>(
            getTypeConverter()->convertType(op.getType())),
        memref, offsets, oneAttrs, oneAttrs);

    rewriter.replaceOp(op, mlir::ValueRange{subview});
    return mlir::success();
  }
};

} // namespace

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([&](mlir::Type ty) -> std::optional<mlir::Type> {
    if (mlir::MemRefType::isValidElementType(ty))
      return ty;
    return std::nullopt;
  });
  converter.addConversion(
      [&](fir::ReferenceType ty) -> std::optional<mlir::Type> {
        auto eleTy = ty.getElementType();
        if (auto sequenceTy = mlir::dyn_cast<fir::SequenceType>(eleTy))
          eleTy = sequenceTy.getElementType();

        if (!mlir::MemRefType::isValidElementType(eleTy))
          return std::nullopt;

        auto layout = mlir::StridedLayoutAttr::get(
            ty.getContext(), mlir::ShapedType::kDynamic, {});
        return mlir::MemRefType::get({}, eleTy, layout);
      });

  // Use fir.convert as the bridge so that we don't need to pull in patterns for
  // other dialects.
  auto materializeProcedure = [](mlir::OpBuilder &builder, mlir::Type type,
                                 mlir::ValueRange inputs,
                                 mlir::Location loc) -> mlir::Value {
    auto convertOp = fir::ConvertOp::create(builder, loc, type, inputs);
    return convertOp;
  };

  converter.addSourceMaterialization(materializeProcedure);
  converter.addTargetMaterialization(materializeProcedure);
  return converter;
}

void FIRToCoreMLIRPass::runOnOperation() {
  mlir::MLIRContext *ctx = &getContext();
  mlir::ModuleOp theModule = getOperation();
  mlir::TypeConverter converter = prepareTypeConverter();
  mlir::RewritePatternSet patterns(ctx);

  patterns.add<FIRAllocOpLowering, FIRLoadOpLowering, FIRStoreOpLowering,
               FIRConvertOpLowering, FIRXArrayCoorOpLowering>(converter, ctx);

  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::arith::ArithDialect, mlir::affine::AffineDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();

  if (mlir::failed(mlir::applyPartialConversion(theModule, target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}
