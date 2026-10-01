//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// C wrappers for libc unittests
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_LIBCCTEST_H
#define LLVM_LIBC_TEST_UNITTEST_LIBCCTEST_H

#include "include/__llvm-libc-common.h"

__BEGIN_C_DECLS

// These symbols are implementation details of the test macros. Do not reference
// them directly.
void libc_c_test_fail(const char *cond, int expected, const char *file,
                      int line);
extern const char LIBC_C_TEST_NAME[];
void libc_c_test_run(void);
void libc_c_test_anchor(void);

__END_C_DECLS

// Use LibcTest.h in C++ code.
#ifndef __cplusplus
#define TEST(name)                                                             \
  const char LIBC_C_TEST_NAME[] = "LlvmLibcCTest." #name;                      \
  void libc_c_test_run_impl(void);                                             \
  void libc_c_test_run(void) {                                                 \
    libc_c_test_anchor(); /* Force a reference to the C test framework. */     \
    libc_c_test_run_impl();                                                    \
  }                                                                            \
  void libc_c_test_run_impl(void)
#define LIBC_C_TEST_SCAFFOLDING_(cond, expected, ret_or_empty)                 \
  do {                                                                         \
    if (!(cond) != !(expected)) {                                              \
      libc_c_test_fail(#cond, (expected), __FILE__, __LINE__);                 \
      ret_or_empty;                                                            \
    }                                                                          \
  } while (0)

#define EXPECT_TRUE(cond) LIBC_C_TEST_SCAFFOLDING_(cond, 1, )
#define ASSERT_TRUE(cond) LIBC_C_TEST_SCAFFOLDING_(cond, 1, return)

#define EXPECT_FALSE(cond) LIBC_C_TEST_SCAFFOLDING_(cond, 0, )
#define ASSERT_FALSE(cond) LIBC_C_TEST_SCAFFOLDING_(cond, 0, return)
#endif

#endif // LLVM_LIBC_TEST_UNITTEST_LIBCCTEST_H
