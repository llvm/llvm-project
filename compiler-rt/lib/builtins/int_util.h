/* ===-- int_util.h - internal utility functions ----------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 *
 * This file is not part of the interface of this library.
 *
 * This file defines non-inline utilities which are available for use in the
 * library. The function definitions themselves are all contained in int_util.c
 * which will always be compiled into any compiler-rt library.
 *
 * ===-----------------------------------------------------------------------===
 */

#ifndef INT_UTIL_H
#define INT_UTIL_H

/** \brief Trigger a program abort (or panic for kernel code). */
#define compilerrt_abort() __compilerrt_abort_impl(__FILE__, __LINE__, __func__)

NORETURN void __compilerrt_abort_impl(const char *file, int line,
                                      const char *function);

#define COMPILE_TIME_ASSERT(expr) COMPILE_TIME_ASSERT1(expr, __COUNTER__)
#define COMPILE_TIME_ASSERT1(expr, cnt) COMPILE_TIME_ASSERT2(expr, cnt)
#define COMPILE_TIME_ASSERT2(expr, cnt)                                        \
  typedef char ct_assert_##cnt[(expr) ? 1 : -1] UNUSED

#endif /* INT_UTIL_H */
