//===-- Simple checkers for integrations tests ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_INTEGRATION_TEST_TEST_H
#define LLVM_LIBC_UTILS_INTEGRATION_TEST_TEST_H

#include "src/__support/OSUtil/io.h"
#include "src/__support/OSUtil/quick_exit.h"

#define __AS_STRING(val) #val
#define __CHECK(file, line, val, should_exit)                                  \
  if (!(val)) {                                                                \
    __llvm_libc::write_to_stderr(file ":" __AS_STRING(                         \
        line) ": Expected '" #val "' to be true, but is false\n");             \
    if (should_exit)                                                           \
      __llvm_libc::quick_exit(127);                                            \
  }

#define __CHECK_NE(file, line, val, should_exit)                               \
  if ((val)) {                                                                 \
    __llvm_libc::write_to_stderr(file ":" __AS_STRING(                         \
        line) ": Expected '" #val "' to be false, but is true\n");             \
    if (should_exit)                                                           \
      __llvm_libc::quick_exit(127);                                            \
  }

#define EXPECT_TRUE(val) __CHECK(__FILE__, __LINE__, val, false)
#define ASSERT_TRUE(val) __CHECK(__FILE__, __LINE__, val, true)
#define EXPECT_FALSE(val) __CHECK_NE(__FILE__, __LINE__, val, false)
#define ASSERT_FALSE(val) __CHECK_NE(__FILE__, __LINE__, val, true)

#endif // LLVM_LIBC_UTILS_INTEGRATION_TEST_TEST_H
