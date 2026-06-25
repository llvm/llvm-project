//===-- FIRBoxUtils.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRBoxUtils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace fir {

void genDimInfoFromBox(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value box,
                       llvm::SmallVectorImpl<mlir::Value> *lbounds,
                       llvm::SmallVectorImpl<mlir::Value> *extents,
                       llvm::SmallVectorImpl<mlir::Value> *strides) {
  auto boxType = mlir::dyn_cast<fir::BaseBoxType>(box.getType());
  assert(boxType && "must be a box");
  if (!lbounds && !extents && !strides)
    return;

  unsigned rank = fir::getBoxRank(boxType);
  assert(!boxType.isAssumedRank() && "must be an array of known rank");
  mlir::Type idxTy = builder.getIndexType();
  for (unsigned i = 0; i < rank; ++i) {
    mlir::Value dim = mlir::arith::ConstantIndexOp::create(builder, loc, i);
    auto dimInfo =
        fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box, dim);
    if (lbounds)
      lbounds->push_back(dimInfo.getLowerBound());
    if (extents)
      extents->push_back(dimInfo.getExtent());
    if (strides)
      strides->push_back(dimInfo.getByteStride());
  }
}

} // namespace fir
