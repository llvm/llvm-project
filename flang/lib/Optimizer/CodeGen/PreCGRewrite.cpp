//===-- PreCGRewrite.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Builder/Todo.h" // remove when TODO's are done
#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "aiir/IR/Iterators.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

namespace fir {
#define GEN_PASS_DEF_CODEGENREWRITE
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

//===----------------------------------------------------------------------===//
// Codegen rewrite: rewriting of subgraphs of ops
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "flang-codegen-rewrite"

static void populateShape(llvm::SmallVectorImpl<aiir::Value> &vec,
                          fir::ShapeOp shape) {
  vec.append(shape.getExtents().begin(), shape.getExtents().end());
}

// Operands of fir.shape_shift split into two vectors.
static void populateShapeAndShift(llvm::SmallVectorImpl<aiir::Value> &shapeVec,
                                  llvm::SmallVectorImpl<aiir::Value> &shiftVec,
                                  fir::ShapeShiftOp shift) {
  for (auto i = shift.getPairs().begin(), endIter = shift.getPairs().end();
       i != endIter;) {
    shiftVec.push_back(*i++);
    shapeVec.push_back(*i++);
  }
}

static void populateShift(llvm::SmallVectorImpl<aiir::Value> &vec,
                          fir::ShiftOp shift) {
  vec.append(shift.getOrigins().begin(), shift.getOrigins().end());
}

namespace {

/// Convert fir.embox to the extended form where necessary.
///
/// The embox operation can take arguments that specify multidimensional array
/// properties at runtime. These properties may be shared between distinct
/// objects that have the same properties. Before we lower these small DAGs to
/// LLVM-IR, we gather all the information into a single extended operation. For
/// example,
/// ```
/// %1 = fir.shape_shift %4, %5 : (index, index) -> !fir.shapeshift<1>
/// %2 = fir.slice %6, %7, %8 : (index, index, index) -> !fir.slice<1>
/// %3 = fir.embox %0 (%1) [%2] : (!fir.ref<!fir.array<?xi32>>,
/// !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
/// ```
/// can be rewritten as
/// ```
/// %1 = fircg.ext_embox %0(%5) origin %4[%6, %7, %8] :
/// (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index) ->
/// !fir.box<!fir.array<?xi32>>
/// ```
class EmboxConversion : public aiir::OpRewritePattern<fir::EmboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(fir::EmboxOp embox,
                  aiir::PatternRewriter &rewriter) const override {
    // If the embox does not include a shape, then do not convert it
    if (auto shapeVal = embox.getShape())
      return rewriteDynamicShape(embox, rewriter, shapeVal);
    if (aiir::isa<fir::ClassType>(embox.getType()))
      TODO(embox.getLoc(), "embox conversion for fir.class type");
    if (auto boxTy = aiir::dyn_cast<fir::BoxType>(embox.getType()))
      if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(boxTy.getEleTy()))
        if (!seqTy.hasDynamicExtents())
          return rewriteStaticShape(embox, rewriter, seqTy);
    return aiir::failure();
  }

  llvm::LogicalResult rewriteStaticShape(fir::EmboxOp embox,
                                         aiir::PatternRewriter &rewriter,
                                         fir::SequenceType seqTy) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<aiir::Value> shapeOpers;
    auto idxTy = rewriter.getIndexType();
    for (auto ext : seqTy.getShape()) {
      auto iAttr = rewriter.getIndexAttr(ext);
      auto extVal =
          aiir::arith::ConstantOp::create(rewriter, loc, idxTy, iAttr);
      shapeOpers.push_back(extVal);
    }
    auto xbox = fir::cg::XEmboxOp::create(
        rewriter, loc, embox.getType(), embox.getMemref(), shapeOpers,
        aiir::ValueRange{}, aiir::ValueRange{}, aiir::ValueRange{},
        aiir::ValueRange{}, embox.getTypeparams(), embox.getSourceBox(),
        embox.getAllocatorIdxAttr());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return aiir::success();
  }

  llvm::LogicalResult rewriteDynamicShape(fir::EmboxOp embox,
                                          aiir::PatternRewriter &rewriter,
                                          aiir::Value shapeVal) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<aiir::Value> shapeOpers;
    llvm::SmallVector<aiir::Value> shiftOpers;
    if (auto shapeOp = aiir::dyn_cast<fir::ShapeOp>(shapeVal.getDefiningOp())) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp =
          aiir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp());
      assert(shiftOp && "shape is neither fir.shape nor fir.shape_shift");
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    llvm::SmallVector<aiir::Value> sliceOpers;
    llvm::SmallVector<aiir::Value> subcompOpers;
    llvm::SmallVector<aiir::Value> substrOpers;
    if (auto s = embox.getSlice())
      if (auto sliceOp =
              aiir::dyn_cast_or_null<fir::SliceOp>(s.getDefiningOp())) {
        sliceOpers.assign(sliceOp.getTriples().begin(),
                          sliceOp.getTriples().end());
        subcompOpers.assign(sliceOp.getFields().begin(),
                            sliceOp.getFields().end());
        substrOpers.assign(sliceOp.getSubstr().begin(),
                           sliceOp.getSubstr().end());
      }
    auto xbox = fir::cg::XEmboxOp::create(
        rewriter, loc, embox.getType(), embox.getMemref(), shapeOpers,
        shiftOpers, sliceOpers, subcompOpers, substrOpers,
        embox.getTypeparams(), embox.getSourceBox(),
        embox.getAllocatorIdxAttr());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return aiir::success();
  }
};

/// Convert fir.rebox to the extended form where necessary.
///
/// For example,
/// ```
/// %5 = fir.rebox %3(%1) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) ->
/// !fir.box<!fir.array<?xi32>>
/// ```
/// converted to
/// ```
/// %5 = fircg.ext_rebox %3(%13) origin %12 : (!fir.box<!fir.array<?xi32>>,
/// index, index) -> !fir.box<!fir.array<?xi32>>
/// ```
class ReboxConversion : public aiir::OpRewritePattern<fir::ReboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(fir::ReboxOp rebox,
                  aiir::PatternRewriter &rewriter) const override {
    auto loc = rebox.getLoc();
    llvm::SmallVector<aiir::Value> shapeOpers;
    llvm::SmallVector<aiir::Value> shiftOpers;
    if (auto shapeVal = rebox.getShape()) {
      if (auto shapeOp = aiir::dyn_cast<fir::ShapeOp>(shapeVal.getDefiningOp()))
        populateShape(shapeOpers, shapeOp);
      else if (auto shiftOp =
                   aiir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp()))
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      else if (auto shiftOp =
                   aiir::dyn_cast<fir::ShiftOp>(shapeVal.getDefiningOp()))
        populateShift(shiftOpers, shiftOp);
      else
        return aiir::failure();
    }
    llvm::SmallVector<aiir::Value> sliceOpers;
    llvm::SmallVector<aiir::Value> subcompOpers;
    llvm::SmallVector<aiir::Value> substrOpers;
    if (auto s = rebox.getSlice())
      if (auto sliceOp =
              aiir::dyn_cast_or_null<fir::SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.getTriples().begin(),
                          sliceOp.getTriples().end());
        subcompOpers.append(sliceOp.getFields().begin(),
                            sliceOp.getFields().end());
        substrOpers.append(sliceOp.getSubstr().begin(),
                           sliceOp.getSubstr().end());
      }

    auto xRebox = fir::cg::XReboxOp::create(
        rewriter, loc, rebox.getType(), rebox.getBox(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, substrOpers);
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << rebox << " to " << xRebox << '\n');
    rewriter.replaceOp(rebox, xRebox.getOperation()->getResults());
    return aiir::success();
  }
};

/// Convert all fir.array_coor to the extended form.
///
/// For example,
/// ```
///  %4 = fir.array_coor %addr (%1) [%2] %0 : (!fir.ref<!fir.array<?xi32>>,
///  !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
/// ```
/// converted to
/// ```
/// %40 = fircg.ext_array_coor %addr(%9) origin %8[%4, %5, %6<%39> :
/// (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index, index) ->
/// !fir.ref<i32>
/// ```
class ArrayCoorConversion : public aiir::OpRewritePattern<fir::ArrayCoorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(fir::ArrayCoorOp arrCoor,
                  aiir::PatternRewriter &rewriter) const override {
    auto loc = arrCoor.getLoc();
    llvm::SmallVector<aiir::Value> shapeOpers;
    llvm::SmallVector<aiir::Value> shiftOpers;
    if (auto shapeVal = arrCoor.getShape()) {
      if (auto shapeOp = aiir::dyn_cast<fir::ShapeOp>(shapeVal.getDefiningOp()))
        populateShape(shapeOpers, shapeOp);
      else if (auto shiftOp =
                   aiir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp()))
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      else if (auto shiftOp =
                   aiir::dyn_cast<fir::ShiftOp>(shapeVal.getDefiningOp()))
        populateShift(shiftOpers, shiftOp);
      else
        return aiir::failure();
    }
    llvm::SmallVector<aiir::Value> sliceOpers;
    llvm::SmallVector<aiir::Value> subcompOpers;
    if (auto s = arrCoor.getSlice())
      if (auto sliceOp =
              aiir::dyn_cast_or_null<fir::SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.getTriples().begin(),
                          sliceOp.getTriples().end());
        subcompOpers.append(sliceOp.getFields().begin(),
                            sliceOp.getFields().end());
        assert(sliceOp.getSubstr().empty() &&
               "Don't allow substring operations on array_coor. This "
               "restriction may be lifted in the future.");
      }
    auto xArrCoor = fir::cg::XArrayCoorOp::create(
        rewriter, loc, arrCoor.getType(), arrCoor.getMemref(), shapeOpers,
        shiftOpers, sliceOpers, subcompOpers, arrCoor.getIndices(),
        arrCoor.getTypeparams());
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << arrCoor << " to " << xArrCoor << '\n');
    rewriter.replaceOp(arrCoor, xArrCoor.getOperation()->getResults());
    return aiir::success();
  }
};

class DeclareOpConversion : public aiir::OpRewritePattern<fir::DeclareOp> {
  bool preserveDeclare;

public:
  using OpRewritePattern::OpRewritePattern;
  DeclareOpConversion(aiir::AIIRContext *ctx, bool preserveDecl)
      : OpRewritePattern(ctx), preserveDeclare(preserveDecl) {}

  llvm::LogicalResult
  matchAndRewrite(fir::DeclareOp declareOp,
                  aiir::PatternRewriter &rewriter) const override {
    if (!preserveDeclare) {
      rewriter.replaceOp(declareOp, declareOp.getMemref());
      return aiir::success();
    }
    auto loc = declareOp.getLoc();
    llvm::SmallVector<aiir::Value> shapeOpers;
    llvm::SmallVector<aiir::Value> shiftOpers;
    if (auto shapeVal = declareOp.getShape()) {
      if (auto shapeOp = aiir::dyn_cast<fir::ShapeOp>(shapeVal.getDefiningOp()))
        populateShape(shapeOpers, shapeOp);
      else if (auto shiftOp =
                   aiir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp()))
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      else if (auto shiftOp =
                   aiir::dyn_cast<fir::ShiftOp>(shapeVal.getDefiningOp()))
        populateShift(shiftOpers, shiftOp);
      else
        return aiir::failure();
    }
    // Extract dummy_arg_no attribute if present
    aiir::IntegerAttr dummyArgNoAttr;
    if (auto attr = declareOp->getAttrOfType<aiir::IntegerAttr>("dummy_arg_no"))
      dummyArgNoAttr = attr;
    // FIXME: Add FortranAttrs and CudaAttrs
    auto xDeclOp = fir::cg::XDeclareOp::create(
        rewriter, loc, declareOp.getType(), declareOp.getMemref(), shapeOpers,
        shiftOpers, declareOp.getTypeparams(), declareOp.getDummyScope(),
        declareOp.getStorage(), declareOp.getStorageOffset(),
        declareOp.getUniqName(), dummyArgNoAttr);
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << declareOp << " to " << xDeclOp << '\n');
    rewriter.replaceOp(declareOp, xDeclOp.getOperation()->getResults());
    return aiir::success();
  }
};

class DummyScopeOpConversion
    : public aiir::OpRewritePattern<fir::DummyScopeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(fir::DummyScopeOp dummyScopeOp,
                  aiir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<fir::UndefOp>(dummyScopeOp,
                                              dummyScopeOp.getType());
    return aiir::success();
  }
};

/// Simple DCE to erase fir.shape/shift/slice/unused shape operands after this
/// pass (fir.shape and like have no codegen).
/// aiir::RegionDCE is expensive and requires running
/// aiir::eraseUnreachableBlocks. It does things that are not needed here, like
/// removing unused block arguments. fir.shape/shift/slice cannot be block
/// arguments.
/// This helper does a naive backward walk of the IR. It is not even guaranteed
/// to walk blocks according to backward dominance, but that is good enough for
/// what is done here, fir.shape/shift/slice have no usages anymore. The
/// backward walk allows getting rid of most of the unused operands, it is not a
/// problem to leave some in the weird cases.
static void simpleDCE(aiir::RewriterBase &rewriter, aiir::Operation *op) {
  op->walk<aiir::WalkOrder::PostOrder, aiir::ReverseIterator>(
      [&](aiir::Operation *subOp) {
        if (aiir::isOpTriviallyDead(subOp))
          rewriter.eraseOp(subOp);
      });
}

class CodeGenRewrite : public fir::impl::CodeGenRewriteBase<CodeGenRewrite> {
public:
  using CodeGenRewriteBase<CodeGenRewrite>::CodeGenRewriteBase;

  void runOnOperation() override final {
    aiir::ModuleOp mod = getOperation();

    auto &context = getContext();
    aiir::ConversionTarget target(context);
    target.addLegalDialect<aiir::arith::ArithDialect, fir::FIROpsDialect,
                           fir::FIRCodeGenDialect, aiir::func::FuncDialect>();
    target.addIllegalOp<fir::ArrayCoorOp>();
    target.addIllegalOp<fir::ReboxOp>();
    target.addIllegalOp<fir::DeclareOp>();
    target.addIllegalOp<fir::DummyScopeOp>();
    target.addDynamicallyLegalOp<fir::EmboxOp>([](fir::EmboxOp embox) {
      return !(embox.getShape() ||
               aiir::isa<fir::SequenceType>(
                   aiir::cast<fir::BaseBoxType>(embox.getType()).getEleTy()));
    });
    aiir::RewritePatternSet patterns(&context);
    fir::populatePreCGRewritePatterns(patterns, preserveDeclare);
    if (aiir::failed(
            aiir::applyPartialConversion(mod, target, std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(&context),
                      "error in running the pre-codegen conversions");
      signalPassFailure();
      return;
    }
    // Erase any residual (fir.shape, fir.slice...).
    aiir::IRRewriter rewriter(&context);
    simpleDCE(rewriter, mod.getOperation());
  }
};

} // namespace

void fir::populatePreCGRewritePatterns(aiir::RewritePatternSet &patterns,
                                       bool preserveDeclare) {
  patterns.insert<EmboxConversion, ArrayCoorConversion, ReboxConversion,
                  DummyScopeOpConversion>(patterns.getContext());
  patterns.add<DeclareOpConversion>(patterns.getContext(), preserveDeclare);
}
