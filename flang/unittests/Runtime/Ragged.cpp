//===-- flang/unittests/Runtime/Ragged.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/ragged.h"
#include "gtest/gtest.h"

using namespace Fortran::runtime;

TEST(Ragged, RaggedArrayAllocateDeallocateTest) {
  struct RaggedArrayHeader header;
  unsigned rank = 2;
  int64_t *extents = new int64_t[2];
  extents[0] = 10;
  extents[1] = 100;
  RaggedArrayHeader *ret = (RaggedArrayHeader *)_FortranARaggedArrayAllocate(
      &header, false, rank, 32, extents);
  EXPECT_TRUE(ret != nullptr);
  EXPECT_TRUE(ret->bufferPointer != nullptr);
  EXPECT_EQ(extents, ret->extentPointer);
  EXPECT_EQ(10, ret->extentPointer[0]);
  EXPECT_EQ(100, ret->extentPointer[1]);
  EXPECT_EQ(rank, ret->getRank());
  EXPECT_FALSE(ret->isIndirection());

  _FortranARaggedArrayDeallocate(ret);
  EXPECT_EQ(0u, ret->getRank());
  EXPECT_FALSE(ret->isIndirection());
}
