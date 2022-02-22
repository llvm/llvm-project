//===-- Unittests for vector ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/vector.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcVectorTest, SimpleConstructor) {
  __llvm_libc::cpp::vector<int> vec;
}

TEST(LlvmLibcVectorTest, OrderedWriteOrderedReadTest) {
  __llvm_libc::cpp::vector<size_t> vec;

  for (size_t i = 0; i < 100; i = i + 2) {
    vec.push_back(i);
  }
  ASSERT_EQ(vec.size(), size_t(50));
  ASSERT_GE(vec.capacity(), vec.size());
  for (size_t j = 0; j < vec.size(); ++j) {
    ASSERT_EQ(vec[j], j * 2);
  }
}

TEST(LlvmLibcVectorTest, ReserveTest) {
  __llvm_libc::cpp::vector<bool> vec;

  size_t prev_capacity = vec.capacity();

  vec.reserve(prev_capacity * 2);

  ASSERT_GT(vec.capacity(), prev_capacity * 2);
}
