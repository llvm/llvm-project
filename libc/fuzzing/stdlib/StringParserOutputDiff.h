//===-- Template to diff single-input-single-output functions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_FUZZING_STDLIB_STRING_PARSER_OUTPUT_DIFF_H
#define LLVM_LIBC_FUZZING_STDLIB_STRING_PARSER_OUTPUT_DIFF_H

#include "fuzzing/math/Compare.h"

#include <stddef.h>
#include <stdint.h>

template <typename T> using StringInputSingleOutputFunc = T (*)(const char *);

template <typename T>
void StringParserOutputDiff(StringInputSingleOutputFunc<T> func1,
                            StringInputSingleOutputFunc<T> func2,
                            const uint8_t *data, size_t size) {
  if (size < sizeof(T))
    return;

  const char *x = reinterpret_cast<const char *>(data);

  T result1 = func1(x);
  T result2 = func2(x);

  if (!ValuesEqual(result1, result2))
    __builtin_trap();
}

#endif // LLVM_LIBC_FUZZING_STDLIB_STRING_PARSER_OUTPUT_DIFF_H
