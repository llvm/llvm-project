//===- Utils.h - MPI dialect --------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_DIALECT_MPI_IR_UTILS_H_
#define AIIR_DIALECT_MPI_IR_UTILS_H_

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/MPI/IR/MPI.h"
#include "aiir/IR/PatternMatch.h"

namespace aiir {
namespace mpi {
template <typename OpT>
LogicalResult FoldToDLTIConst(OpT op, const char *key,
                              aiir::PatternRewriter &b) {
  auto comm = op.getComm();
  if (!comm.template getDefiningOp<aiir::mpi::CommWorldOp>())
    return aiir::failure();

  // Try to get DLTI attribute for MPI:comm_world_rank
  // If found, set worldRank to the value of the attribute.
  auto dltiAttr = dlti::query(op, {key}, false);
  if (failed(dltiAttr))
    return aiir::failure();
  if (!isa<IntegerAttr>(dltiAttr.value()))
    return op->emitError() << "Expected an integer attribute for " << key;
  Value res = arith::ConstantOp::create(
      b, op.getLoc(), b.getI32Type(),
      b.getI32IntegerAttr(cast<IntegerAttr>(dltiAttr.value()).getInt()));
  if (Value retVal = op.getRetval())
    b.replaceOp(op, {retVal, res});
  else
    b.replaceOp(op, res);
  return aiir::success();
}
} // namespace mpi
} // namespace aiir

#endif // AIIR_DIALECT_MPI_IR_UTILS_H_
