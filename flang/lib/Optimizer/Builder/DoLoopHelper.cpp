//===-- DoLoopHelper.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/DoLoopHelper.h"

//===----------------------------------------------------------------------===//
// DoLoopHelper implementation
//===----------------------------------------------------------------------===//

fir::DoLoopOp
fir::factory::DoLoopHelper::createLoop(mlir::Value lb, mlir::Value ub,
                                       mlir::Value step,
                                       const BodyGenerator &bodyGenerator) {
  auto lbi = builder.convertToIndexType(loc, lb);
  auto ubi = builder.convertToIndexType(loc, ub);
  assert(step && "step must be an actual Value");
  auto inc = builder.convertToIndexType(loc, step);
  auto loop = builder.create<fir::DoLoopOp>(loc, lbi, ubi, inc);
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(loop.getBody());
  auto index = loop.getInductionVar();
  bodyGenerator(builder, index);
  builder.restoreInsertionPoint(insertPt);
  return loop;
}

fir::DoLoopOp
fir::factory::DoLoopHelper::createLoop(mlir::Value lb, mlir::Value ub,
                                       const BodyGenerator &bodyGenerator) {
  return createLoop(
      lb, ub, builder.createIntegerConstant(loc, builder.getIndexType(), 1),
      bodyGenerator);
}

fir::DoLoopOp
fir::factory::DoLoopHelper::createLoop(mlir::Value count,
                                       const BodyGenerator &bodyGenerator) {
  auto indexType = builder.getIndexType();
  auto zero = builder.createIntegerConstant(loc, indexType, 0);
  auto one = builder.createIntegerConstant(loc, count.getType(), 1);
  auto up = builder.create<arith::SubIOp>(loc, count, one);
  return createLoop(zero, up, one, bodyGenerator);
}
