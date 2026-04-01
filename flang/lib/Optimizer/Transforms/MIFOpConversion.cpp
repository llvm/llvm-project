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
#include "flang/Runtime/stop.h"
#include "aiir/IR/Matchers.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_MIFOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace aiir;
using namespace Fortran::runtime;

namespace {

// Default prefix for subroutines of PRIF compiled with LLVM
static std::string getPRIFProcName(std::string fmt) {
  std::ostringstream oss;
  oss << "prif_" << fmt;
  return fir::NameUniquer::doProcedure({"prif"}, {}, oss.str());
}

static aiir::Type getPRIFStatType(fir::FirOpBuilder &builder) {
  return builder.getRefType(builder.getI32Type());
}

static aiir::Type getPRIFErrmsgType(fir::FirOpBuilder &builder) {
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
static std::pair<aiir::Value, aiir::Value>
genErrmsgPRIF(fir::FirOpBuilder &builder, aiir::Location loc,
              aiir::Value errmsg) {
  aiir::Value absent =
      fir::AbsentOp::create(builder, loc, getPRIFErrmsgType(builder));
  if (!errmsg)
    return {absent, absent};

  bool isAllocatableErrmsg = fir::isAllocatableType(errmsg.getType());
  aiir::Value errMsg = isAllocatableErrmsg ? absent : errmsg;
  aiir::Value errMsgAlloc = isAllocatableErrmsg ? errmsg : absent;
  return {errMsg, errMsgAlloc};
}

static aiir::Value genStatPRIF(fir::FirOpBuilder &builder, aiir::Location loc,
                               aiir::Value stat) {
  if (!stat)
    return fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
  return stat;
}

static fir::CallOp genPRIFStopErrorStop(fir::FirOpBuilder &builder,
                                        aiir::Location loc,
                                        aiir::Value stopCode,
                                        bool isError = false) {
  aiir::Type stopCharTy = fir::BoxCharType::get(builder.getContext(), 1);
  aiir::Type i1Ty = builder.getI1Type();
  aiir::Type i32Ty = builder.getI32Type();

  aiir::FunctionType ftype = aiir::FunctionType::get(
      builder.getContext(),
      /*inputs*/
      {builder.getRefType(i1Ty), builder.getRefType(i32Ty), stopCharTy},
      /*results*/ {});
  aiir::func::FuncOp funcOp =
      isError
          ? builder.createFunction(loc, getPRIFProcName("error_stop"), ftype)
          : builder.createFunction(loc, getPRIFProcName("stop"), ftype);

  // QUIET is managed in flang-rt, so its value is set to TRUE here.
  aiir::Value q = builder.createBool(loc, true);
  aiir::Value quiet = builder.createTemporary(loc, i1Ty);
  fir::StoreOp::create(builder, loc, q, quiet);

  aiir::Value stopCodeInt, stopCodeChar;
  if (!stopCode) {
    stopCodeChar = fir::AbsentOp::create(builder, loc, stopCharTy);
    stopCodeInt =
        fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));
  } else if (fir::isa_integer(stopCode.getType())) {
    stopCodeChar = fir::AbsentOp::create(builder, loc, stopCharTy);
    stopCodeInt = builder.createTemporary(loc, i32Ty);
    if (stopCode.getType() != i32Ty)
      stopCode = fir::ConvertOp::create(builder, loc, i32Ty, stopCode);
    fir::StoreOp::create(builder, loc, stopCode, stopCodeInt);
  } else {
    stopCodeChar = stopCode;
    if (!aiir::isa<fir::BoxCharType>(stopCodeChar.getType())) {
      auto len =
          fir::UndefOp::create(builder, loc, builder.getCharacterLengthType());
      stopCodeChar =
          fir::EmboxCharOp::create(builder, loc, stopCharTy, stopCodeChar, len);
    }
    stopCodeInt =
        fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));
  }

  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, ftype, quiet, stopCodeInt, stopCodeChar);
  return fir::CallOp::create(builder, loc, funcOp, args);
}

enum class TerminationKind { Normal = 0, Error = 1, FailImage = 2 };
// Generates a wrapper function for the different kind of termination in PRIF.
// This function will be used to register wrappers on PRIF runtime termination
// functions into the Fortran runtime.
aiir::Value genTerminationOperationWrapper(fir::FirOpBuilder &builder,
                                           aiir::Location loc,
                                           aiir::ModuleOp module,
                                           TerminationKind termKind) {
  std::string funcName;
  aiir::FunctionType funcType =
      aiir::FunctionType::get(builder.getContext(), {}, {});
  aiir::Type i32Ty = builder.getI32Type();
  if (termKind == TerminationKind::Normal) {
    funcName = getPRIFProcName("stop");
    funcType = aiir::FunctionType::get(builder.getContext(), {i32Ty}, {});
  } else if (termKind == TerminationKind::Error) {
    funcName = getPRIFProcName("error_stop");
    funcType = aiir::FunctionType::get(builder.getContext(), {i32Ty}, {});
  } else {
    funcName = getPRIFProcName("fail_image");
  }
  funcName += "_termination_wrapper";
  aiir::func::FuncOp funcWrapperOp =
      module.lookupSymbol<aiir::func::FuncOp>(funcName);

  if (!funcWrapperOp) {
    funcWrapperOp = builder.createFunction(loc, funcName, funcType);

    // generating the body of the function.
    aiir::OpBuilder::InsertPoint saveInsertPoint = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(funcWrapperOp.addEntryBlock());

    if (termKind == TerminationKind::Normal) {
      genPRIFStopErrorStop(builder, loc, funcWrapperOp.getArgument(0),
                           /*isError*/ false);
    } else if (termKind == TerminationKind::Error) {
      genPRIFStopErrorStop(builder, loc, funcWrapperOp.getArgument(0),
                           /*isError*/ true);
    } else {
      aiir::func::FuncOp fOp = builder.createFunction(
          loc, getPRIFProcName("fail_image"),
          aiir::FunctionType::get(builder.getContext(), {}, {}));
      fir::CallOp::create(builder, loc, fOp);
    }

    aiir::func::ReturnOp::create(builder, loc);
    builder.restoreInsertionPoint(saveInsertPoint);
  }

  aiir::SymbolRefAttr symbolRef = aiir::SymbolRefAttr::get(
      builder.getContext(), funcWrapperOp.getSymNameAttr());
  return fir::AddrOfOp::create(builder, loc, funcType, symbolRef);
}

/// Convert mif.init operation to runtime call of 'prif_init'
struct MIFInitOpConversion : public aiir::OpRewritePattern<mif::InitOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::InitOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    aiir::Type i32Ty = builder.getI32Type();
    aiir::Value result = builder.createTemporary(loc, i32Ty);

    // Registering PRIF runtime termination to the Fortran runtime
    // STOP
    aiir::Value funcStopOp = genTerminationOperationWrapper(
        builder, loc, mod, TerminationKind::Normal);
    aiir::func::FuncOp normalEndFunc =
        fir::runtime::getRuntimeFunc<mkRTKey(RegisterImagesNormalEndCallback)>(
            loc, builder);
    llvm::SmallVector<aiir::Value> args1 = fir::runtime::createArguments(
        builder, loc, normalEndFunc.getFunctionType(), funcStopOp);
    fir::CallOp::create(builder, loc, normalEndFunc, args1);

    // ERROR STOP
    aiir::Value funcErrorStopOp = genTerminationOperationWrapper(
        builder, loc, mod, TerminationKind::Error);
    aiir::func::FuncOp errorFunc =
        fir::runtime::getRuntimeFunc<mkRTKey(RegisterImagesErrorCallback)>(
            loc, builder);
    llvm::SmallVector<aiir::Value> args2 = fir::runtime::createArguments(
        builder, loc, errorFunc.getFunctionType(), funcErrorStopOp);
    fir::CallOp::create(builder, loc, errorFunc, args2);

    // FAIL IMAGE
    aiir::Value failImageOp = genTerminationOperationWrapper(
        builder, loc, mod, TerminationKind::FailImage);
    aiir::func::FuncOp failImageFunc =
        fir::runtime::getRuntimeFunc<mkRTKey(RegisterFailImageCallback)>(
            loc, builder);
    llvm::SmallVector<aiir::Value> args3 = fir::runtime::createArguments(
        builder, loc, errorFunc.getFunctionType(), failImageOp);
    fir::CallOp::create(builder, loc, failImageFunc, args3);

    // Intialize the multi-image parallel environment
    aiir::FunctionType ftype = aiir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {builder.getRefType(i32Ty)}, /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("init"), ftype);
    llvm::SmallVector<aiir::Value> args =
        fir::runtime::createArguments(builder, loc, ftype, result);
    fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOpWithNewOp<fir::LoadOp>(op, result);
    return aiir::success();
  }
};

/// Convert mif.this_image operation to PRIF runtime call
struct MIFThisImageOpConversion
    : public aiir::OpRewritePattern<mif::ThisImageOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::ThisImageOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    if (op.getCoarray())
      TODO(loc, "mif.this_image op with coarray argument.");
    else {
      aiir::Type i32Ty = builder.getI32Type();
      aiir::Type boxTy = fir::BoxType::get(rewriter.getNoneType());
      aiir::Value result = builder.createTemporary(loc, i32Ty);
      aiir::FunctionType ftype = aiir::FunctionType::get(
          builder.getContext(),
          /*inputs*/ {boxTy, builder.getRefType(i32Ty)}, /*results*/ {});
      aiir::Value teamArg = op.getTeam();
      if (!op.getTeam())
        teamArg = fir::AbsentOp::create(builder, loc, boxTy);

      aiir::func::FuncOp funcOp = builder.createFunction(
          loc, getPRIFProcName("this_image_no_coarray"), ftype);
      llvm::SmallVector<aiir::Value> args =
          fir::runtime::createArguments(builder, loc, ftype, teamArg, result);
      fir::CallOp::create(builder, loc, funcOp, args);
      rewriter.replaceOpWithNewOp<fir::LoadOp>(op, result);
      return aiir::success();
    }
  }
};

/// Convert mif.num_images operation to runtime call of
/// prif_num_images_with_{team|team_number}
struct MIFNumImagesOpConversion
    : public aiir::OpRewritePattern<mif::NumImagesOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::NumImagesOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    aiir::Type i32Ty = builder.getI32Type();
    aiir::Type i64Ty = builder.getI64Type();
    aiir::Type boxTy = fir::BoxType::get(rewriter.getNoneType());
    aiir::Value result = builder.createTemporary(loc, i32Ty);

    aiir::func::FuncOp funcOp;
    llvm::SmallVector<aiir::Value> args;
    if (!op.getTeam() && !op.getTeamNumber()) {
      aiir::FunctionType ftype = aiir::FunctionType::get(
          builder.getContext(),
          /*inputs*/ {builder.getRefType(i32Ty)}, /*results*/ {});
      funcOp =
          builder.createFunction(loc, getPRIFProcName("num_images"), ftype);
      args = fir::runtime::createArguments(builder, loc, ftype, result);
    } else {
      if (op.getTeam()) {
        aiir::FunctionType ftype =
            aiir::FunctionType::get(builder.getContext(),
                                    /*inputs*/
                                    {boxTy, builder.getRefType(i32Ty)},
                                    /*results*/ {});
        funcOp = builder.createFunction(
            loc, getPRIFProcName("num_images_with_team"), ftype);
        args = fir::runtime::createArguments(builder, loc, ftype, op.getTeam(),
                                             result);
      } else {
        aiir::Value teamNumber = builder.createTemporary(loc, i64Ty);
        aiir::Value cst = op.getTeamNumber();
        if (op.getTeamNumber().getType() != i64Ty)
          cst = fir::ConvertOp::create(builder, loc, i64Ty, op.getTeamNumber());
        fir::StoreOp::create(builder, loc, cst, teamNumber);
        aiir::FunctionType ftype = aiir::FunctionType::get(
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
    return aiir::success();
  }
};

/// Convert mif.sync_all operation to runtime call of 'prif_sync_all'
struct MIFSyncAllOpConversion : public aiir::OpRewritePattern<mif::SyncAllOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::SyncAllOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    aiir::Type errmsgTy = getPRIFErrmsgType(builder);
    aiir::FunctionType ftype = aiir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_all"), ftype);

    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    aiir::Value stat = genStatPRIF(builder, loc, op.getStat());
    llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return aiir::success();
  }
};

/// Convert mif.sync_images operation to runtime call of 'prif_sync_images'
struct MIFSyncImagesOpConversion
    : public aiir::OpRewritePattern<mif::SyncImagesOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::SyncImagesOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    aiir::Type errmsgTy = getPRIFErrmsgType(builder);
    aiir::Type imgSetTy = fir::BoxType::get(fir::SequenceType::get(
        {fir::SequenceType::getUnknownExtent()}, builder.getI32Type()));
    aiir::FunctionType ftype = aiir::FunctionType::get(
        builder.getContext(),
        /*inputs*/
        {imgSetTy, getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_images"), ftype);

    // If imageSet is scalar, PRIF require to pass an array of size 1.
    aiir::Value imageSet = op.getImageSet();
    if (!imageSet)
      imageSet = fir::AbsentOp::create(builder, loc, imgSetTy);
    else if (auto boxTy = aiir::dyn_cast<fir::BoxType>(imageSet.getType())) {
      if (!aiir::isa<fir::SequenceType>(boxTy.getEleTy())) {
        aiir::Value one =
            builder.createIntegerConstant(loc, builder.getI32Type(), 1);
        aiir::Value shape = fir::ShapeOp::create(builder, loc, one);
        imageSet =
            fir::ReboxOp::create(builder, loc,
                                 fir::BoxType::get(fir::SequenceType::get(
                                     {1}, builder.getI32Type())),
                                 imageSet, shape, aiir::Value{});
      }
    }
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    aiir::Value stat = genStatPRIF(builder, loc, op.getStat());
    llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, imageSet, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return aiir::success();
  }
};

/// Convert mif.sync_memory operation to runtime call of 'prif_sync_memory'
struct MIFSyncMemoryOpConversion
    : public aiir::OpRewritePattern<mif::SyncMemoryOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::SyncMemoryOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    aiir::Type errmsgTy = getPRIFErrmsgType(builder);
    aiir::FunctionType ftype = aiir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_memory"), ftype);

    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    aiir::Value stat = genStatPRIF(builder, loc, op.getStat());
    llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return aiir::success();
  }
};

/// Convert mif.sync_team operation to runtime call of 'prif_sync_team'
struct MIFSyncTeamOpConversion
    : public aiir::OpRewritePattern<mif::SyncTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::SyncTeamOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    aiir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    aiir::Type errmsgTy = getPRIFErrmsgType(builder);
    aiir::FunctionType ftype = aiir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {boxTy, getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_team"), ftype);

    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    aiir::Value stat = genStatPRIF(builder, loc, op.getStat());
    llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, op.getTeam(), stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return aiir::success();
  }
};

/// Generate call to collective subroutines except co_reduce
/// A must be lowered as a box
static fir::CallOp genCollectiveSubroutine(fir::FirOpBuilder &builder,
                                           aiir::Location loc, aiir::Value A,
                                           aiir::Value image, aiir::Value stat,
                                           aiir::Value errmsg,
                                           std::string coName) {
  aiir::Value rootImage;
  aiir::Type i32Ty = builder.getI32Type();
  if (!image)
    rootImage = fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));
  else {
    rootImage = builder.createTemporary(loc, i32Ty);
    if (image.getType() != i32Ty)
      image = fir::ConvertOp::create(builder, loc, i32Ty, image);
    fir::StoreOp::create(builder, loc, image, rootImage);
  }

  aiir::Type errmsgTy = getPRIFErrmsgType(builder);
  aiir::Type boxTy = fir::BoxType::get(builder.getNoneType());
  aiir::FunctionType ftype =
      aiir::FunctionType::get(builder.getContext(),
                              /*inputs*/
                              {boxTy, builder.getRefType(builder.getI32Type()),
                               getPRIFStatType(builder), errmsgTy, errmsgTy},
                              /*results*/ {});
  aiir::func::FuncOp funcOp = builder.createFunction(loc, coName, ftype);

  auto [errmsgArg, errmsgAllocArg] = genErrmsgPRIF(builder, loc, errmsg);
  if (!stat)
    stat = fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, ftype, A, rootImage, stat, errmsgArg, errmsgAllocArg);
  return fir::CallOp::create(builder, loc, funcOp, args);
}

/// Convert mif.co_broadcast operation to runtime call of 'prif_co_broadcast'
struct MIFCoBroadcastOpConversion
    : public aiir::OpRewritePattern<mif::CoBroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::CoBroadcastOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    fir::CallOp callOp = genCollectiveSubroutine(
        builder, loc, op.getA(), op.getSourceImage(), op.getStat(),
        op.getErrmsg(), getPRIFProcName("co_broadcast"));
    rewriter.replaceOp(op, callOp);
    return aiir::success();
  }
};

/// Convert mif.co_max operation to runtime call of 'prif_co_max'
struct MIFCoMaxOpConversion : public aiir::OpRewritePattern<mif::CoMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::CoMaxOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    fir::CallOp callOp;
    aiir::Type argTy =
        fir::unwrapSequenceType(fir::unwrapPassByRefType(op.getA().getType()));
    if (aiir::isa<fir::CharacterType>(argTy))
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_max_character"));
    else
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_max"));
    rewriter.replaceOp(op, callOp);
    return aiir::success();
  }
};

/// Convert mif.co_min operation to runtime call of 'prif_co_min'
struct MIFCoMinOpConversion : public aiir::OpRewritePattern<mif::CoMinOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::CoMinOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    fir::CallOp callOp;
    aiir::Type argTy =
        fir::unwrapSequenceType(fir::unwrapPassByRefType(op.getA().getType()));
    if (aiir::isa<fir::CharacterType>(argTy))
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_min_character"));
    else
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_min"));
    rewriter.replaceOp(op, callOp);
    return aiir::success();
  }
};

/// Convert mif.co_sum operation to runtime call of 'prif_co_sum'
struct MIFCoSumOpConversion : public aiir::OpRewritePattern<mif::CoSumOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::CoSumOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    fir::CallOp callOp = genCollectiveSubroutine(
        builder, loc, op.getA(), op.getResultImage(), op.getStat(),
        op.getErrmsg(), getPRIFProcName("co_sum"));
    rewriter.replaceOp(op, callOp);
    return aiir::success();
  }
};

/// Convert mif.form_team operation to runtime call of 'prif_form_team'
struct MIFFormTeamOpConversion
    : public aiir::OpRewritePattern<mif::FormTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::FormTeamOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();
    aiir::Type errmsgTy = getPRIFErrmsgType(builder);
    aiir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    aiir::FunctionType ftype = aiir::FunctionType::get(
        builder.getContext(),
        /*inputs*/
        {builder.getRefType(builder.getI64Type()), boxTy,
         builder.getRefType(builder.getI32Type()), getPRIFStatType(builder),
         errmsgTy, errmsgTy},
        /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("form_team"), ftype);

    aiir::Type i64Ty = builder.getI64Type();
    aiir::Value teamNumber = builder.createTemporary(loc, i64Ty);
    aiir::Value t =
        (op.getTeamNumber().getType() == i64Ty)
            ? op.getTeamNumber()
            : fir::ConvertOp::create(builder, loc, i64Ty, op.getTeamNumber());
    fir::StoreOp::create(builder, loc, t, teamNumber);

    aiir::Type i32Ty = builder.getI32Type();
    aiir::Value newIndex;
    if (op.getNewIndex()) {
      newIndex = builder.createTemporary(loc, i32Ty);
      aiir::Value ni =
          (op.getNewIndex().getType() == i32Ty)
              ? op.getNewIndex()
              : fir::ConvertOp::create(builder, loc, i32Ty, op.getNewIndex());
      fir::StoreOp::create(builder, loc, ni, newIndex);
    } else
      newIndex = fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));

    aiir::Value stat = genStatPRIF(builder, loc, op.getStat());
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, teamNumber, op.getTeamVar(), newIndex, stat,
        errmsgArg, errmsgAllocArg);
    fir::CallOp callOp = fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOp(op, callOp);
    return aiir::success();
  }
};

/// Convert mif.change_team operation to runtime call of 'prif_change_team'
struct MIFChangeTeamOpConversion
    : public aiir::OpRewritePattern<mif::ChangeTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::ChangeTeamOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    builder.setInsertionPoint(op);

    aiir::Location loc = op.getLoc();
    aiir::Type errmsgTy = getPRIFErrmsgType(builder);
    aiir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    aiir::FunctionType ftype = aiir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {boxTy, getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("change_team"), ftype);

    aiir::Value stat = genStatPRIF(builder, loc, op.getStat());
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, op.getTeam(), stat, errmsgArg, errmsgAllocArg);
    fir::CallOp::create(builder, loc, funcOp, args);

    aiir::Operation *changeOp = op.getOperation();
    auto &bodyRegion = op.getRegion();
    aiir::Block &bodyBlock = bodyRegion.front();

    rewriter.inlineBlockBefore(&bodyBlock, changeOp);
    rewriter.eraseOp(op);
    return aiir::success();
  }
};

/// Convert mif.end_team operation to runtime call of 'prif_end_team'
struct MIFEndTeamOpConversion : public aiir::OpRewritePattern<mif::EndTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::EndTeamOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();
    aiir::Type errmsgTy = getPRIFErrmsgType(builder);
    aiir::FunctionType ftype = aiir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("end_team"), ftype);

    aiir::Value stat = genStatPRIF(builder, loc, op.getStat());
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, stat, errmsgArg, errmsgAllocArg);
    fir::CallOp callOp = fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOp(op, callOp);
    return aiir::success();
  }
};

/// Convert mif.get_team operation to runtime call of 'prif_get_team'
struct MIFGetTeamOpConversion : public aiir::OpRewritePattern<mif::GetTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::GetTeamOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();

    aiir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    aiir::Type lvlTy = builder.getRefType(builder.getI32Type());
    aiir::FunctionType ftype =
        aiir::FunctionType::get(builder.getContext(),
                                /*inputs*/ {lvlTy, boxTy},
                                /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("get_team"), ftype);

    aiir::Value level = op.getLevel();
    if (!level)
      level = fir::AbsentOp::create(builder, loc, lvlTy);
    else {
      aiir::Value cst = op.getLevel();
      aiir::Type i32Ty = builder.getI32Type();
      level = builder.createTemporary(loc, i32Ty);
      if (cst.getType() != i32Ty)
        cst = builder.createConvert(loc, i32Ty, cst);
      fir::StoreOp::create(builder, loc, cst, level);
    }
    aiir::Type resultType = op.getResult().getType();
    aiir::Type baseTy = fir::unwrapRefType(resultType);
    aiir::Value team = builder.createTemporary(loc, baseTy);
    fir::EmboxOp box = fir::EmboxOp::create(builder, loc, resultType, team);

    llvm::SmallVector<aiir::Value> args =
        fir::runtime::createArguments(builder, loc, ftype, level, box);
    fir::CallOp::create(builder, loc, funcOp, args);

    rewriter.replaceOp(op, box);
    return aiir::success();
  }
};

/// Convert mif.team_number operation to runtime call of 'prif_team_number'
struct MIFTeamNumberOpConversion
    : public aiir::OpRewritePattern<mif::TeamNumberOp> {
  using OpRewritePattern::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(mif::TeamNumberOp op,
                  aiir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    aiir::Location loc = op.getLoc();
    aiir::Type i64Ty = builder.getI64Type();
    aiir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    aiir::FunctionType ftype =
        aiir::FunctionType::get(builder.getContext(),
                                /*inputs*/ {boxTy, builder.getRefType(i64Ty)},
                                /*results*/ {});
    aiir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("team_number"), ftype);

    aiir::Value team = op.getTeam();
    if (!team)
      team = fir::AbsentOp::create(builder, loc, boxTy);

    aiir::Value result = builder.createTemporary(loc, i64Ty);
    llvm::SmallVector<aiir::Value> args =
        fir::runtime::createArguments(builder, loc, ftype, team, result);
    fir::CallOp::create(builder, loc, funcOp, args);
    fir::LoadOp load = fir::LoadOp::create(builder, loc, result);
    rewriter.replaceOp(op, load);
    return aiir::success();
  }
};

class MIFOpConversion : public fir::impl::MIFOpConversionBase<MIFOpConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    aiir::RewritePatternSet patterns(ctx);
    aiir::ConversionTarget target(*ctx);

    mif::populateMIFOpConversionPatterns(patterns);

    target.addLegalDialect<fir::FIROpsDialect>();
    target.addLegalOp<aiir::ModuleOp>();

    if (aiir::failed(aiir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(ctx),
                      "error in MIF op conversion\n");
      return signalPassFailure();
    }
  }
};
} // namespace

void mif::populateMIFOpConversionPatterns(aiir::RewritePatternSet &patterns) {
  patterns.insert<MIFInitOpConversion, MIFThisImageOpConversion,
                  MIFNumImagesOpConversion, MIFSyncAllOpConversion,
                  MIFSyncImagesOpConversion, MIFSyncMemoryOpConversion,
                  MIFSyncTeamOpConversion, MIFCoBroadcastOpConversion,
                  MIFCoMaxOpConversion, MIFCoMinOpConversion,
                  MIFCoSumOpConversion, MIFFormTeamOpConversion,
                  MIFChangeTeamOpConversion, MIFEndTeamOpConversion,
                  MIFGetTeamOpConversion, MIFTeamNumberOpConversion>(
      patterns.getContext());
}
