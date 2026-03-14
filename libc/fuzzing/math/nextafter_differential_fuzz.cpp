//===-- nextafter_differential_fuzz.cpp
//---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Differential fuzz test for llvm-libc nextafter implementation.
///
//===----------------------------------------------------------------------===//

#include "fuzzing/math/TwoInputSingleOutputDiff.h"

#include "src/math/nextafter.h"
#include "src/math/nextafterf.h"
#include "src/math/nextafterl.h"

#include <math.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  TwoInputSingleOutputDiff<float, float>(&LIBC_NAMESPACE::nextafterf,
                                         &::nextafterf, data, size);
  TwoInputSingleOutputDiff<double, double>(&LIBC_NAMESPACE::nextafter,
                                           &::nextafter, data, size);
  return 0;
}
