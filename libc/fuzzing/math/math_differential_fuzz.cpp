//===-- ldexp_differential_fuzz.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Differential fuzz test for llvm-libc ldexp implementation.
///
//===----------------------------------------------------------------------===//

#include "fuzzing/math/RemQuoDiff.h"
#include "fuzzing/math/SingleInputSingleOutputDiff.h"
#include "fuzzing/math/TwoInputSingleOutputDiff.h"

#include "src/math/ceil.h"
#include "src/math/ceilf.h"
#include "src/math/ceill.h"

#include "src/math/fdim.h"
#include "src/math/fdimf.h"
#include "src/math/fdiml.h"

#include "src/math/floor.h"
#include "src/math/floorf.h"
#include "src/math/floorl.h"

#include "src/math/frexp.h"
#include "src/math/frexpf.h"
#include "src/math/frexpl.h"

#include "src/math/hypotf.h"

#include "src/math/ldexp.h"
#include "src/math/ldexpf.h"
#include "src/math/ldexpl.h"

#include "src/math/logb.h"
#include "src/math/logbf.h"
#include "src/math/logbl.h"

#include "src/math/modf.h"
#include "src/math/modff.h"
#include "src/math/modfl.h"

#include "src/math/remainder.h"
#include "src/math/remainderf.h"
#include "src/math/remainderl.h"

#include "src/math/remquo.h"
#include "src/math/remquof.h"
#include "src/math/remquol.h"

#include "src/math/round.h"
#include "src/math/roundf.h"
#include "src/math/roundl.h"

#include "src/math/sqrt.h"
#include "src/math/sqrtf.h"
#include "src/math/sqrtl.h"

#include "src/math/trunc.h"
#include "src/math/truncf.h"
#include "src/math/truncl.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {

  SingleInputSingleOutputDiff<float>(&__llvm_libc::ceilf, &::ceilf, data, size);
  SingleInputSingleOutputDiff<double>(&__llvm_libc::ceil, &::ceil, data, size);
  SingleInputSingleOutputDiff<long double>(&__llvm_libc::ceill, &::ceill, data,
                                           size);

  SingleInputSingleOutputDiff<float>(&__llvm_libc::floorf, &::floorf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&__llvm_libc::floor, &::floor, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&__llvm_libc::floorl, &::floorl,
                                           data, size);

  SingleInputSingleOutputDiff<float>(&__llvm_libc::roundf, &::roundf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&__llvm_libc::round, &::round, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&__llvm_libc::roundl, &::roundl,
                                           data, size);

  SingleInputSingleOutputDiff<float>(&__llvm_libc::truncf, &::truncf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&__llvm_libc::trunc, &::trunc, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&__llvm_libc::truncl, &::truncl,
                                           data, size);

  SingleInputSingleOutputDiff<float>(&__llvm_libc::logbf, &::logbf, data, size);
  SingleInputSingleOutputDiff<double>(&__llvm_libc::logb, &::logb, data, size);
  SingleInputSingleOutputDiff<long double>(&__llvm_libc::logbl, &::logbl, data,
                                           size);

  TwoInputSingleOutputDiff<float, float>(&__llvm_libc::hypotf, &::hypotf, data,
                                         size);

  TwoInputSingleOutputDiff<float, float>(&__llvm_libc::remainderf,
                                         &::remainderf, data, size);
  TwoInputSingleOutputDiff<double, double>(&__llvm_libc::remainder,
                                           &::remainder, data, size);
  TwoInputSingleOutputDiff<long double, long double>(&__llvm_libc::remainderl,
                                                     &::remainderl, data, size);

  TwoInputSingleOutputDiff<float, float>(&__llvm_libc::fdimf, &::fdimf, data,
                                         size);
  TwoInputSingleOutputDiff<double, double>(&__llvm_libc::fdim, &::fdim, data,
                                           size);
  TwoInputSingleOutputDiff<long double, long double>(&__llvm_libc::fdiml,
                                                     &::fdiml, data, size);

  SingleInputSingleOutputDiff<float>(&__llvm_libc::sqrtf, &::sqrtf, data, size);
  SingleInputSingleOutputDiff<double>(&__llvm_libc::sqrt, &::sqrt, data, size);
  SingleInputSingleOutputDiff<long double>(&__llvm_libc::sqrtl, &::sqrtl, data,
                                           size);

  SingleInputSingleOutputWithSideEffectDiff<float, int>(&__llvm_libc::frexpf,
                                                        &::frexpf, data, size);
  SingleInputSingleOutputWithSideEffectDiff<double, int>(&__llvm_libc::frexp,
                                                         &::frexp, data, size);
  SingleInputSingleOutputWithSideEffectDiff<long double, int>(
      &__llvm_libc::frexpl, &::frexpl, data, size);

  SingleInputSingleOutputWithSideEffectDiff<float, float>(&__llvm_libc::modff,
                                                          &::modff, data, size);
  SingleInputSingleOutputWithSideEffectDiff<double, double>(
      &__llvm_libc::modf, &::modf, data, size);
  SingleInputSingleOutputWithSideEffectDiff<long double, long double>(
      &__llvm_libc::modfl, &::modfl, data, size);

  TwoInputSingleOutputDiff<float, int>(&__llvm_libc::ldexpf, &::ldexpf, data,
                                       size);
  TwoInputSingleOutputDiff<double, int>(&__llvm_libc::ldexp, &::ldexp, data,
                                        size);
  TwoInputSingleOutputDiff<long double, int>(&__llvm_libc::ldexpl, &::ldexpl,
                                             data, size);

  RemQuoDiff<float>(&__llvm_libc::remquof, &::remquof, data, size);
  RemQuoDiff<double>(&__llvm_libc::remquo, &::remquo, data, size);
  RemQuoDiff<long double>(&__llvm_libc::remquol, &::remquol, data, size);

  return 0;
}
