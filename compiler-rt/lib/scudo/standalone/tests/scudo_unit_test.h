//===-- scudo_unit_test.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

#if SCUDO_FUCHSIA
#include <zxtest/zxtest.h>
#else
#include "gtest/gtest.h"
#endif

// If EXPECT_DEATH isn't defined, make it a no-op.
#ifndef EXPECT_DEATH
#define EXPECT_DEATH(X, Y)                                                     \
  do {                                                                         \
  } while (0)
#endif

// If EXPECT_STREQ isn't defined, define our own simple one.
#ifndef EXPECT_STREQ
#define EXPECT_STREQ(X, Y) EXPECT_EQ(strcmp(X, Y), 0)
#endif

extern bool UseQuarantine;
