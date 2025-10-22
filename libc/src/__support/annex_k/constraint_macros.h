//===-- Helper macros header for constraint violations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ANNEX_K_MACROS_H
#define LLVM_LIBC_SRC___SUPPORT_ANNEX_K_MACROS_H

#include "libc_constraint_handler.h"
#include "src/__support/libc_errno.h"

#define _CONSTRAINT_VIOLATION(msg, error_code, ret_code)                       \
  {                                                                            \
    libc_errno = error_code;                                                   \
    libc_constraint_handler(msg, nullptr, error_code);                         \
    return ret_code;                                                           \
  }

#define _CONSTRAINT_VIOLATION_IF(expr, error_code, return_code)                \
  {                                                                            \
    auto expr_val = expr;                                                      \
    if (expr_val) {                                                            \
      libc_errno = error_code;                                                 \
      libc_constraint_handler(nullptr, nullptr, error_code);                   \
      return return_code;                                                      \
    }                                                                          \
  }

#define _CONSTRAINT_VIOLATION_CLEANUP_IF(expr, cleanup, error_code,            \
                                         return_code)                          \
  {                                                                            \
    auto expr_val = expr;                                                      \
    if (expr_val) {                                                            \
      cleanup;                                                                 \
      libc_errno = error_code;                                                 \
      libc_constraint_handler(nullptr, nullptr, error_code);                   \
      return return_code;                                                      \
    }                                                                          \
  }

#endif // LLVM_LIBC_SRC___SUPPORT_ANNEX_K_MACROS_H
