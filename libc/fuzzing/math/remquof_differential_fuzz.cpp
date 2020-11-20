//===-- remquof_differential_fuzz.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Differential fuzz test for llvm-libc remquof implementation.
///
//===----------------------------------------------------------------------===//

#include "fuzzing/math/RemQuoDiff.h"
#include "src/math/remquof.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  RemQuoDiff<float>(&__llvm_libc::remquof, &::remquof, data, size);
  return 0;
}
