//===- StructuredOpsUtilsTest.cpp - StructuredOpsUtils unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Utils/StructuredOpsUtils.h"
#include "aiir/IR/AffineExpr.h"
#include "aiir/IR/AffineMap.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace aiir;
using testing::Not;
using testing::Truly;

namespace {

TEST(isRowMajorMatmul, Simple) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorMatmul));
}

TEST(isRowMajorMatmul, BindingShifted) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, k, m, n); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorMatmul));
}

TEST(isRowMajorMatmul, BindingSwapped) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, k, n, m); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorMatmul));
}

TEST(isRowMajorMatmul, ColumnMajor) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isRowMajorMatmul, FirstInputSwapped) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, m}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isRowMajorMatmul, TooFewMaps) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isRowMajorMatmul, TooManyMaps) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto mapD = AffineMapAttr::get(AffineMap::get(3, 0, {k, m}, &context));

  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC, mapD});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isRowMajorMatmul, TooFewOutputs) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isColumnMajorMatmul, Simple) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isColumnMajorMatmul));
}

TEST(isColumnMajorMatmul, BindingShifted) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, k, m, n); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isColumnMajorMatmul));
}

TEST(isColumnMajorMatmul, BindingSwapped) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, k, n, m); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isColumnMajorMatmul));
}

TEST(isColumnMajorMatmul, RowMajor) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isColumnMajorMatmul)));
}

TEST(isColumnMajorMatmul, FirstInputSwapped) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {n, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isColumnMajorMatmul)));
}

TEST(isRowMajorBatchMatmul, Simple) {
  AIIRContext context;

  AffineExpr batch, m, n, k;
  bindDims(&context, batch, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorBatchMatmul));
}

TEST(isRowMajorBatchMatmul, BindingShifted) {
  AIIRContext context;

  AffineExpr batch, m, n, k;
  bindDims(&context, k, batch, m, n); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorBatchMatmul));
}

TEST(isRowMajorBatchMatmul, BindingSwapped) {
  AIIRContext context;

  AffineExpr batch, m, n, k;
  bindDims(&context, batch, k, n, m); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorBatchMatmul));
}

TEST(isRowMajorBatchMatmul, FirstInputSwapped) {
  AIIRContext context;

  AffineExpr batch, m, n, k;
  bindDims(&context, batch, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, m}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isRowMajorBatchMatmul)));
}

TEST(isVecmat, Simple) {
  AIIRContext context;

  AffineExpr k, n;
  bindDims(&context, k, n);
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isVecmat));
}

TEST(isVecmat, BindingSwapped) {
  AIIRContext context;

  AffineExpr k, n;
  bindDims(&context, n, k); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isVecmat));
}

TEST(isVecmat, WrongDimOrderMatrix) {
  AIIRContext context;

  AffineExpr k, n;
  bindDims(&context, k, n);
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {n, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isVecmat)));
}

TEST(isMatvec, Simple) {
  AIIRContext context;

  AffineExpr k, n;
  bindDims(&context, k, n);
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {n, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isMatvec));
}

TEST(isMatvec, BindingSwapped) {
  AIIRContext context;

  AffineExpr k, n;
  bindDims(&context, n, k); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {n, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isMatvec));
}

TEST(isMatvec, WrongDimOrderMatrix) {
  AIIRContext context;

  AffineExpr k, n;
  bindDims(&context, k, n);
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isMatvec)));
}

TEST(isBatchMatvec, Simple) {
  AIIRContext context;

  AffineExpr batch, k, n;
  bindDims(&context, batch, k, n);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isBatchMatvec));
}

TEST(isBatchMatvec, BindingSwapped) {
  AIIRContext context;

  AffineExpr batch, k, n;
  bindDims(&context, batch, n, k); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isBatchMatvec));
}

TEST(isBatchMatvec, Matmul) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isBatchMatvec)));
}

TEST(isBatchMatvec, WrongDimOrderMatrix) {
  AIIRContext context;

  AffineExpr batch, k, n;
  bindDims(&context, batch, k, n);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isBatchMatvec)));
}

TEST(isBatchVecmat, Simple) {
  AIIRContext context;

  AffineExpr batch, k, n;
  bindDims(&context, batch, k, n);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isBatchVecmat));
}

TEST(isBatchVecmat, BindingSwapped) {
  AIIRContext context;

  AffineExpr batch, k, n;
  bindDims(&context, batch, n, k); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isBatchVecmat));
}

TEST(isBatchVecmat, Matmul) {
  AIIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isBatchVecmat)));
}

TEST(isBatchVecmat, WrongDimOrderMatrix) {
  AIIRContext context;

  AffineExpr batch, k, n;
  bindDims(&context, batch, k, n);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {batch, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {batch, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isBatchVecmat)));
}

} // namespace
