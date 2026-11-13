//===-- Unittests for the pthread_once_t type -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for pthread_once_t being an integer, as it is in glibc, musl and
/// bionic.
///
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/pthread-macros.h"
#include "include/llvm-libc-types/pthread_once_t.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcPthreadOnceTypeTest, IsAnInt) {
  static_assert(__is_same(pthread_once_t, int), "pthread_once_t is an int");
  EXPECT_EQ(sizeof(pthread_once_t), sizeof(int));
}

TEST(LlvmLibcPthreadOnceTypeTest, InitialiserIsAnIntegerConstant) {
  pthread_once_t once = PTHREAD_ONCE_INIT;
  EXPECT_TRUE(once == PTHREAD_ONCE_INIT);
  EXPECT_EQ(once, 0);
  once = PTHREAD_ONCE_INIT;
  EXPECT_EQ(once, 0);
}
