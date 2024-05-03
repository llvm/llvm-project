//===-- A data structure for str_to_number to return ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STR_TO_NUM_RESULT_H
#define LLVM_LIBC_SRC___SUPPORT_STR_TO_NUM_RESULT_H

#include "src/__support/macros/attributes.h" // LIBC_INLINE

#include <stddef.h>

namespace LIBC_NAMESPACE {

template <typename T> struct StrToNumResult {
  T value;
  int error;
  ptrdiff_t parsed_len;

  LIBC_INLINE constexpr StrToNumResult(T value)
      : value(value), error(0), parsed_len(0) {}
  LIBC_INLINE constexpr StrToNumResult(T value, ptrdiff_t parsed_len)
      : value(value), error(0), parsed_len(parsed_len) {}
  LIBC_INLINE constexpr StrToNumResult(T value, ptrdiff_t parsed_len, int error)
      : value(value), error(error), parsed_len(parsed_len) {}

  LIBC_INLINE constexpr bool has_error() { return error != 0; }

  LIBC_INLINE constexpr operator T() { return value; }
};
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_STR_TO_NUM_RESULT_H
