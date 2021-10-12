//===-- sanitizer_stackdepot_test.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_stackdepot.h"

#include <regex>

#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"

namespace __sanitizer {

TEST(SanitizerCommon, StackDepotBasic) {
  uptr array[] = {1, 2, 3, 4, 5};
  StackTrace s1(array, ARRAY_SIZE(array));
  u32 i1 = StackDepotPut(s1);
  StackTrace stack = StackDepotGet(i1);
  EXPECT_NE(stack.trace, (uptr*)0);
  EXPECT_EQ(ARRAY_SIZE(array), stack.size);
  EXPECT_EQ(0, internal_memcmp(stack.trace, array, sizeof(array)));
}

TEST(SanitizerCommon, StackDepotAbsent) {
  StackTrace stack = StackDepotGet((1 << 30) - 1);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST(SanitizerCommon, StackDepotEmptyStack) {
  u32 i1 = StackDepotPut(StackTrace());
  StackTrace stack = StackDepotGet(i1);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST(SanitizerCommon, StackDepotZeroId) {
  StackTrace stack = StackDepotGet(0);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST(SanitizerCommon, StackDepotSame) {
  uptr array[] = {1, 2, 3, 4, 6};
  StackTrace s1(array, ARRAY_SIZE(array));
  u32 i1 = StackDepotPut(s1);
  u32 i2 = StackDepotPut(s1);
  EXPECT_EQ(i1, i2);
  StackTrace stack = StackDepotGet(i1);
  EXPECT_NE(stack.trace, (uptr*)0);
  EXPECT_EQ(ARRAY_SIZE(array), stack.size);
  EXPECT_EQ(0, internal_memcmp(stack.trace, array, sizeof(array)));
}

TEST(SanitizerCommon, StackDepotSeveral) {
  uptr array1[] = {1, 2, 3, 4, 7};
  StackTrace s1(array1, ARRAY_SIZE(array1));
  u32 i1 = StackDepotPut(s1);
  uptr array2[] = {1, 2, 3, 4, 8, 9};
  StackTrace s2(array2, ARRAY_SIZE(array2));
  u32 i2 = StackDepotPut(s2);
  EXPECT_NE(i1, i2);
}

TEST(SanitizerCommon, StackDepotPrint) {
  uptr array1[] = {0x111, 0x222, 0x333, 0x444, 0x777};
  StackTrace s1(array1, ARRAY_SIZE(array1));
  u32 i1 = StackDepotPut(s1);
  uptr array2[] = {0x1111, 0x2222, 0x3333, 0x4444, 0x8888, 0x9999};
  StackTrace s2(array2, ARRAY_SIZE(array2));
  u32 i2 = StackDepotPut(s2);
  EXPECT_NE(i1, i2);

  auto fix_regex = [](const std::string& s) -> std::string {
    if (!SANITIZER_WINDOWS)
      return s;
    return std::regex_replace(s, std::regex("\\.\\*"), ".*\\n.*");
  };
  EXPECT_EXIT(
      (StackDepotPrintAll(), exit(0)), ::testing::ExitedWithCode(0),
      fix_regex("Stack for id .*#0 0x1.*#1 0x2.*#2 0x3.*#3 0x4.*#4 0x7.*"));
  EXPECT_EXIT(
      (StackDepotPrintAll(), exit(0)), ::testing::ExitedWithCode(0),
      fix_regex(
          "Stack for id .*#0 0x1.*#1 0x2.*#2 0x3.*#3 0x4.*#4 0x8.*#5 0x9.*"));
}

TEST(SanitizerCommon, StackDepotPrintNoLock) {
  u32 n = 2000;
  std::vector<u32> idx2id(n);
  for (u32 i = 0; i < n; ++i) {
    uptr array[] = {0x111, 0x222, i, 0x444, 0x777};
    StackTrace s(array, ARRAY_SIZE(array));
    idx2id[i] = StackDepotPut(s);
  }
  StackDepotPrintAll();
  for (u32 i = 0; i < n; ++i) {
    uptr array[] = {0x111, 0x222, i, 0x444, 0x777};
    StackTrace s(array, ARRAY_SIZE(array));
    CHECK_EQ(idx2id[i], StackDepotPut(s));
  }
}

}  // namespace __sanitizer
