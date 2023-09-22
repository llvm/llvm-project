//===-- A data structure for str_to_number to return ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STR_TO_NUM_RESULT_H
#define LLVM_LIBC_SRC___SUPPORT_STR_TO_NUM_RESULT_H

#include <stddef.h>

namespace __llvm_libc {

template <typename T> struct StrToNumResult {
  T value;
  int error;
  ptrdiff_t parsed_len;

  constexpr StrToNumResult(T value) : value(value), error(0), parsed_len(0) {}
  constexpr StrToNumResult(T value, ptrdiff_t parsed_len)
      : value(value), error(0), parsed_len(parsed_len) {}
  constexpr StrToNumResult(T value, ptrdiff_t parsed_len, int error)
      : value(value), error(error), parsed_len(parsed_len) {}

  constexpr bool has_error() { return error != 0; }

  constexpr operator T() { return value; }
};
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_STR_TO_NUM_RESULT_H
