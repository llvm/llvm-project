//===-- MIFOpConversion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/MIFOpConversion.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/MIF/MIFOps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_MIFOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;
using namespace Fortran::runtime;

namespace {

// Default prefix for subroutines of PRIF compiled with LLVM
static std::string getPRIFProcName(std::string fmt) {
  std::ostringstream oss;
  oss << "prif_" << fmt;
  return fir::NameUniquer::doProcedure({"prif"}, {}, oss.str());
}

static mlir::Type getPRIFStatType(fir::FirOpBuilder &builder) {
  return builder.getRefType(builder.getI32Type());
}

static mlir::Type getPRIFErrmsgType(fir::FirOpBuilder &builder) {
  return fir::BoxType::get(fir::CharacterType::get(
      builder.getContext(), 1, fir::CharacterType::unknownLen()));
}

// Most PRIF functions take `errmsg` and `errmsg_alloc` as two optional
// arguments of intent (out). One is allocatable, the other is not.
// It is the responsibility of the compiler to ensure that the appropriate
// optional argument is passed, and at most one must be provided in a given
// call.
// Depending on the type of `errmsg`, this function will return the pair
// corresponding to (`errmsg`, `errmsg_alloc`).
static std::pair<mlir::Value, mlir::Value>
genErrmsgPRIF(fir::FirOpBuilder &builder, mlir::Location loc,
              mlir::Value errmsg) {
  mlir::Value absent =
      fir::AbsentOp::create(builder, loc, getPRIFErrmsgType(builder));
  if (!errmsg)
    return {absent, absent};

  bool isAllocatableErrmsg = fir::isAllocatableType(errmsg.getType());
  mlir::Value errMsg = isAllocatableErrmsg ? absent : errmsg;
  mlir::Value errMsgAlloc = isAllocatableErrmsg ? errmsg : absent;
  return {errMsg, errMsgAlloc};
}

/// Convert mif.init operation to runtime call of 'prif_init'
struct MIFInitOpConversion : public mlir::OpRewritePattern<mif::InitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::InitOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type i32Ty = builder.getI32Type();
    mlir::Value result = builder.createTemporary(loc, i32Ty);
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {builder.getRefType(i32Ty)}, /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("init"), ftype);
    llvm::SmallVector<mlir::Value> args =
        fir::runtime::createArguments(builder, loc, ftype, result);
    fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOpWithNewOp<fir::LoadOp>(op, result);
    return mlir::success();
  }
};

/// Convert mif.this_image operation to PRIF runtime call
struct MIFThisImageOpConversion
    : public mlir::OpRewritePattern<mif::ThisImageOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::ThisImageOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    if (op.getCoarray())
      TODO(loc, "mif.this_image op with coarray argument.");
    else {
      mlir::Type i32Ty = builder.getI32Type();
      mlir::Type boxTy = fir::BoxType::get(rewriter.getNoneType());
      mlir::Value result = builder.createTemporary(loc, i32Ty);
      mlir::FunctionType ftype = mlir::FunctionType::get(
          builder.getContext(),
          /*inputs*/ {boxTy, builder.getRefType(i32Ty)}, /*results*/ {});
      mlir::Value teamArg = op.getTeam();
      if (!op.getTeam())
        teamArg = fir::AbsentOp::create(builder, loc, boxTy);

      mlir::func::FuncOp funcOp = builder.createFunction(
          loc, getPRIFProcName("this_image_no_coarray"), ftype);
      llvm::SmallVector<mlir::Value> args =
          fir::runtime::createArguments(builder, loc, ftype, teamArg, result);
      fir::CallOp::create(builder, loc, funcOp, args);
      rewriter.replaceOpWithNewOp<fir::LoadOp>(op, result);
      return mlir::success();
    }
  }
};

/// Convert mif.num_images operation to runtime call of
/// prif_num_images_with_{team|team_number}
struct MIFNumImagesOpConversion
    : public mlir::OpRewritePattern<mif::NumImagesOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::NumImagesOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type i32Ty = builder.getI32Type();
    mlir::Type i64Ty = builder.getI64Type();
    mlir::Type boxTy = fir::BoxType::get(rewriter.getNoneType());
    mlir::Value result = builder.createTemporary(loc, i32Ty);

    mlir::func::FuncOp funcOp;
    llvm::SmallVector<mlir::Value> args;
    if (!op.getTeam() && !op.getTeamNumber()) {
      mlir::FunctionType ftype = mlir::FunctionType::get(
          builder.getContext(),
          /*inputs*/ {builder.getRefType(i32Ty)}, /*results*/ {});
      funcOp =
          builder.createFunction(loc, getPRIFProcName("num_images"), ftype);
      args = fir::runtime::createArguments(builder, loc, ftype, result);
    } else {
      if (op.getTeam()) {
        mlir::FunctionType ftype =
            mlir::FunctionType::get(builder.getContext(),
                                    /*inputs*/
                                    {boxTy, builder.getRefType(i32Ty)},
                                    /*results*/ {});
        funcOp = builder.createFunction(
            loc, getPRIFProcName("num_images_with_team"), ftype);
        args = fir::runtime::createArguments(builder, loc, ftype, op.getTeam(),
                                             result);
      } else {
        mlir::Value teamNumber = builder.createTemporary(loc, i64Ty);
        mlir::Value cst = op.getTeamNumber();
        if (op.getTeamNumber().getType() != i64Ty)
          cst = fir::ConvertOp::create(builder, loc, i64Ty, op.getTeamNumber());
        fir::StoreOp::create(builder, loc, cst, teamNumber);
        mlir::FunctionType ftype = mlir::FunctionType::get(
            builder.getContext(),
            /*inputs*/ {builder.getRefType(i64Ty), builder.getRefType(i32Ty)},
            /*results*/ {});
        funcOp = builder.createFunction(
            loc, getPRIFProcName("num_images_with_team_number"), ftype);
        args = fir::runtime::createArguments(builder, loc, ftype, teamNumber,
                                             result);
      }
    }
    fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOpWithNewOp<fir::LoadOp>(op, result);
    return mlir::success();
  }
};

/// Convert mif.sync_all operation to runtime call of 'prif_sync_all'
struct MIFSyncAllOpConversion : public mlir::OpRewritePattern<mif::SyncAllOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::SyncAllOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_all"), ftype);

    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    mlir::Value stat = op.getStat();
    if (!stat)
      stat = fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

/// Convert mif.sync_images operation to runtime call of 'prif_sync_images'
struct MIFSyncImagesOpConversion
    : public mlir::OpRewritePattern<mif::SyncImagesOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::SyncImagesOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::Type imgSetTy = fir::BoxType::get(fir::SequenceType::get(
        {fir::SequenceType::getUnknownExtent()}, builder.getI32Type()));
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/
        {imgSetTy, getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_images"), ftype);

    // If imageSet is scalar, PRIF require to pass an array of size 1.
    mlir::Value imageSet = op.getImageSet();
    if (!imageSet)
      imageSet = fir::AbsentOp::create(builder, loc, imgSetTy);
    else if (auto boxTy = mlir::dyn_cast<fir::BoxType>(imageSet.getType())) {
      if (!mlir::isa<fir::SequenceType>(boxTy.getEleTy())) {
        mlir::Value one =
            builder.createIntegerConstant(loc, builder.getI32Type(), 1);
        mlir::Value shape = fir::ShapeOp::create(builder, loc, one);
        imageSet =
            fir::ReboxOp::create(builder, loc,
                                 fir::BoxType::get(fir::SequenceType::get(
                                     {1}, builder.getI32Type())),
                                 imageSet, shape, mlir::Value{});
      }
    }
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    mlir::Value stat = op.getStat();
    if (!stat)
      stat = fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, imageSet, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

/// Convert mif.sync_memory operation to runtime call of 'prif_sync_memory'
struct MIFSyncMemoryOpConversion
    : public mlir::OpRewritePattern<mif::SyncMemoryOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::SyncMemoryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_memory"), ftype);

    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    mlir::Value stat = op.getStat();
    if (!stat)
      stat = fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

/// Generate call to collective subroutines except co_reduce
/// A must be lowered as a box
static fir::CallOp genCollectiveSubroutine(fir::FirOpBuilder &builder,
                                           mlir::Location loc, mlir::Value A,
                                           mlir::Value image, mlir::Value stat,
                                           mlir::Value errmsg,
                                           std::string coName) {
  mlir::Value rootImage;
  mlir::Type i32Ty = builder.getI32Type();
  if (!image)
    rootImage = fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));
  else {
    rootImage = builder.createTemporary(loc, i32Ty);
    if (image.getType() != i32Ty)
      image = fir::ConvertOp::create(builder, loc, i32Ty, image);
    fir::StoreOp::create(builder, loc, image, rootImage);
  }

  mlir::Type errmsgTy = getPRIFErrmsgType(builder);
  mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
  mlir::FunctionType ftype =
      mlir::FunctionType::get(builder.getContext(),
                              /*inputs*/
                              {boxTy, builder.getRefType(builder.getI32Type()),
                               getPRIFStatType(builder), errmsgTy, errmsgTy},
                              /*results*/ {});
  mlir::func::FuncOp funcOp = builder.createFunction(loc, coName, ftype);

  auto [errmsgArg, errmsgAllocArg] = genErrmsgPRIF(builder, loc, errmsg);
  if (!stat)
    stat = fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, ftype, A, rootImage, stat, errmsgArg, errmsgAllocArg);
  return fir::CallOp::create(builder, loc, funcOp, args);
}

/// Convert mif.co_broadcast operation to runtime call of 'prif_co_broadcast'
struct MIFCoBroadcastOpConversion
    : public mlir::OpRewritePattern<mif::CoBroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::CoBroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    fir::CallOp callOp = genCollectiveSubroutine(
        builder, loc, op.getA(), op.getSourceImage(), op.getStat(),
        op.getErrmsg(), getPRIFProcName("co_broadcast"));
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.co_max operation to runtime call of 'prif_co_max'
struct MIFCoMaxOpConversion : public mlir::OpRewritePattern<mif::CoMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::CoMaxOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    fir::CallOp callOp;
    mlir::Type argTy =
        fir::unwrapSequenceType(fir::unwrapPassByRefType(op.getA().getType()));
    if (mlir::isa<fir::CharacterType>(argTy))
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_max_character"));
    else
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_max"));
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.co_min operation to runtime call of 'prif_co_min'
struct MIFCoMinOpConversion : public mlir::OpRewritePattern<mif::CoMinOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::CoMinOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    fir::CallOp callOp;
    mlir::Type argTy =
        fir::unwrapSequenceType(fir::unwrapPassByRefType(op.getA().getType()));
    if (mlir::isa<fir::CharacterType>(argTy))
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_min_character"));
    else
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_min"));
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.co_sum operation to runtime call of 'prif_co_sum'
struct MIFCoSumOpConversion : public mlir::OpRewritePattern<mif::CoSumOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::CoSumOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    fir::CallOp callOp = genCollectiveSubroutine(
        builder, loc, op.getA(), op.getResultImage(), op.getStat(),
        op.getErrmsg(), getPRIFProcName("co_sum"));
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

class MIFOpConversion : public fir::impl::MIFOpConversionBase<MIFOpConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);

    mif::populateMIFOpConversionPatterns(patterns);

    target.addLegalDialect<fir::FIROpsDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in MIF op conversion\n");
      return signalPassFailure();
    }
  }
};
} // namespace

void mif::populateMIFOpConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<MIFInitOpConversion, MIFThisImageOpConversion,
                  MIFNumImagesOpConversion, MIFSyncAllOpConversion,
                  MIFSyncImagesOpConversion, MIFSyncMemoryOpConversion,
                  MIFCoBroadcastOpConversion, MIFCoMaxOpConversion,
                  MIFCoMinOpConversion, MIFCoSumOpConversion>(
      patterns.getContext());
}
