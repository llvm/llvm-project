//===-- remquo_differential_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Differential fuzz test for llvm-libc remquo implementation.
///
//===----------------------------------------------------------------------===//

#include "fuzzing/math/RemQuoDiff.h"
#include "src/math/remquo.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  RemQuoDiff<double>(&__llvm_libc::remquo, &::remquo, data, size);
  return 0;
}
