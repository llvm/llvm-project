//===- MPIOps.cpp - MPI dialect ops implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MPI/IR/MPI.h"
#include "aiir/Dialect/MPI/IR/Utils.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinAttributes.h"

using namespace aiir;
using namespace aiir::mpi;

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult aiir::mpi::ReduceScatterBlockOp::verify() {
  if (getSendbuf().getType().getElementType() !=
      getRecvbuf().getType().getElementType())
    return emitOpError("sendbuf and recvbuf must have the same element type");
  return success();
}

namespace {

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

// If input memref has dynamic shape and is a cast and if the cast's input has
// static shape, fold the cast's static input into the given operation.
template <typename OpT>
struct FoldCast final : public aiir::OpRewritePattern<OpT> {
  using aiir::OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                aiir::PatternRewriter &b) const override {
    auto mRef = op.getRef();
    if (mRef.getType().hasStaticShape()) {
      return aiir::failure();
    }
    auto defOp = mRef.getDefiningOp();
    if (!defOp || !aiir::isa<aiir::memref::CastOp>(defOp)) {
      return aiir::failure();
    }
    auto src = aiir::cast<aiir::memref::CastOp>(defOp).getSource();
    if (!src.getType().hasStaticShape()) {
      return aiir::failure();
    }
    b.modifyOpInPlace(op, [&]() { op.getRefMutable().assign(src); });
    return aiir::success();
  }
};

struct FoldRank final : public aiir::OpRewritePattern<aiir::mpi::CommRankOp> {
  using aiir::OpRewritePattern<aiir::mpi::CommRankOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(aiir::mpi::CommRankOp op,
                                aiir::PatternRewriter &b) const override {
    return FoldToDLTIConst(op, "MPI:comm_world_rank", b);
  }
};

struct FoldSize final : public aiir::OpRewritePattern<aiir::mpi::CommSizeOp> {
  using aiir::OpRewritePattern<aiir::mpi::CommSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(aiir::mpi::CommSizeOp op,
                                aiir::PatternRewriter &b) const override {
    return FoldToDLTIConst(op, "MPI:comm_world_size", b);
  }
};
} // namespace

void aiir::mpi::SendOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &results, aiir::AIIRContext *context) {
  results.add<FoldCast<aiir::mpi::SendOp>>(context);
}

void aiir::mpi::RecvOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &results, aiir::AIIRContext *context) {
  results.add<FoldCast<aiir::mpi::RecvOp>>(context);
}

void aiir::mpi::ISendOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &results, aiir::AIIRContext *context) {
  results.add<FoldCast<aiir::mpi::ISendOp>>(context);
}

void aiir::mpi::IRecvOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &results, aiir::AIIRContext *context) {
  results.add<FoldCast<aiir::mpi::IRecvOp>>(context);
}

void aiir::mpi::CommRankOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &results, aiir::AIIRContext *context) {
  results.add<FoldRank>(context);
}

void aiir::mpi::CommSizeOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &results, aiir::AIIRContext *context) {
  results.add<FoldSize>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/MPI/IR/MPIOps.cpp.inc"
