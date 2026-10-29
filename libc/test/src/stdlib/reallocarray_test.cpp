//===-- Unittests for reallocarray ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/func/free.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/libc_errno.h"
#include "src/stdlib/reallocarray.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcReallocarrayTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcReallocarrayTest, AllocAndGrow) {
  constexpr size_t INITIAL_COUNT = 8;
  int *arr = reinterpret_cast<int *>(
      LIBC_NAMESPACE::reallocarray(nullptr, INITIAL_COUNT, sizeof(int)));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(arr, static_cast<int *>(nullptr));

  for (size_t i = 0; i < INITIAL_COUNT; ++i)
    arr[i] = static_cast<int>(i * 3 + 1);

  constexpr size_t EXPANDED_COUNT = 16;
  arr = reinterpret_cast<int *>(
      LIBC_NAMESPACE::reallocarray(arr, EXPANDED_COUNT, sizeof(int)));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(arr, static_cast<int *>(nullptr));

  for (size_t i = 0; i < INITIAL_COUNT; ++i)
    EXPECT_EQ(arr[i], static_cast<int>(i * 3 + 1));

  ::free(arr);
}

TEST_F(LlvmLibcReallocarrayTest, OverflowFails) {
  constexpr size_t SIZE_MAX_VAL = ~size_t(0);

  libc_errno = 0;
  void *res = LIBC_NAMESPACE::reallocarray(nullptr, SIZE_MAX_VAL, 2);
  EXPECT_EQ(res, static_cast<void *>(nullptr));
  ASSERT_ERRNO_EQ(ENOMEM);

  libc_errno = 0;
  res = LIBC_NAMESPACE::reallocarray(nullptr, 2, SIZE_MAX_VAL / 2 + 1);
  EXPECT_EQ(res, static_cast<void *>(nullptr));
  ASSERT_ERRNO_EQ(ENOMEM);

  libc_errno = 0;
  res = LIBC_NAMESPACE::reallocarray(nullptr, SIZE_MAX_VAL, SIZE_MAX_VAL);
  EXPECT_EQ(res, static_cast<void *>(nullptr));
  ASSERT_ERRNO_EQ(ENOMEM);
}

TEST_F(LlvmLibcReallocarrayTest, ZeroSize) {
  void *res = LIBC_NAMESPACE::reallocarray(nullptr, 0, sizeof(int));
  ::free(res);

  res = LIBC_NAMESPACE::reallocarray(nullptr, 10, 0);
  ::free(res);
}
