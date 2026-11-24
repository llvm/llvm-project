//===- unittests/AST/APValueTest.cpp - APValue tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"
#include "llvm/ADT/DenseMap.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <utility>

using namespace clang;

namespace {

static APValue makeInt(uint64_t Value) {
  return APValue(llvm::APSInt(llvm::APInt(32, Value), /*IsUnsigned=*/false));
}

static APValue makeNestedValue() {
  APValue VectorElements[] = {makeInt(1), makeInt(2)};
  APValue Vector(VectorElements, 2);

  APValue MatrixElements[] = {makeInt(3), makeInt(4), makeInt(5), makeInt(6)};
  APValue Matrix(MatrixElements, 2, 2);

  APValue Base(APValue::UninitStruct(), 0, 2);
  Base.getStructField(0) = std::move(Vector);
  Base.getStructField(1) = std::move(Matrix);

  APValue Array(APValue::UninitArray(), 1, 2);
  Array.getArrayInitializedElt(0) = makeInt(7);
  Array.getArrayFiller() = makeInt(0);

  APValue Union(static_cast<const FieldDecl *>(nullptr));
  APValue VirtualBase(APValue::UninitStruct(), 0, 1);
  VirtualBase.getStructField(0) = std::move(Union);

  APValue Result(APValue::UninitStruct(), 1, 1, 1);
  Result.getStructBase(0) = std::move(Base);
  Result.getStructField(0) = std::move(Array);
  Result.getStructVirtualBase(0) = std::move(VirtualBase);
  return Result;
}

TEST(APValueTest, VisitCountsNestedValueKinds) {
  APValue Value = makeNestedValue();
  llvm::DenseMap<APValue::ValueKind, unsigned> Counts;
  Value.visit([&](const APValue &SubValue) {
    ++Counts[SubValue.getKind()];
    return true;
  });

  const llvm::DenseMap<APValue::ValueKind, unsigned> Expected = {
      {APValue::Int, 8},   {APValue::Vector, 1}, {APValue::Matrix, 1},
      {APValue::Array, 1}, {APValue::Struct, 3}, {APValue::Union, 1},
  };
  EXPECT_EQ(Counts, Expected);
}

TEST(APValueTest, VisitStopsWhenCallbackReturnsFalse) {
  APValue Value = makeNestedValue();
  for (unsigned StopAfter : {1u, 2u, 5u}) {
    unsigned NumVisited = 0;
    Value.visit([&](const APValue &) { return ++NumVisited < StopAfter; });
    EXPECT_EQ(NumVisited, StopAfter);
  }
}

} // namespace
