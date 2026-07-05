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

  SingleInputSingleOutputDiff<float>(&LIBC_NAMESPACE::ceilf, &::ceilf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&LIBC_NAMESPACE::ceil, &::ceil, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&LIBC_NAMESPACE::ceill, &::ceill,
                                           data, size);

  SingleInputSingleOutputDiff<float>(&LIBC_NAMESPACE::floorf, &::floorf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&LIBC_NAMESPACE::floor, &::floor, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&LIBC_NAMESPACE::floorl, &::floorl,
                                           data, size);

  SingleInputSingleOutputDiff<float>(&LIBC_NAMESPACE::roundf, &::roundf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&LIBC_NAMESPACE::round, &::round, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&LIBC_NAMESPACE::roundl, &::roundl,
                                           data, size);

  SingleInputSingleOutputDiff<float>(&LIBC_NAMESPACE::truncf, &::truncf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&LIBC_NAMESPACE::trunc, &::trunc, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&LIBC_NAMESPACE::truncl, &::truncl,
                                           data, size);

  SingleInputSingleOutputDiff<float>(&LIBC_NAMESPACE::logbf, &::logbf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&LIBC_NAMESPACE::logb, &::logb, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&LIBC_NAMESPACE::logbl, &::logbl,
                                           data, size);

  TwoInputSingleOutputDiff<float, float>(&LIBC_NAMESPACE::hypotf, &::hypotf,
                                         data, size);

  TwoInputSingleOutputDiff<float, float>(&LIBC_NAMESPACE::remainderf,
                                         &::remainderf, data, size);
  TwoInputSingleOutputDiff<double, double>(&LIBC_NAMESPACE::remainder,
                                           &::remainder, data, size);
  TwoInputSingleOutputDiff<long double, long double>(
      &LIBC_NAMESPACE::remainderl, &::remainderl, data, size);

  TwoInputSingleOutputDiff<float, float>(&LIBC_NAMESPACE::fdimf, &::fdimf, data,
                                         size);
  TwoInputSingleOutputDiff<double, double>(&LIBC_NAMESPACE::fdim, &::fdim, data,
                                           size);
  TwoInputSingleOutputDiff<long double, long double>(&LIBC_NAMESPACE::fdiml,
                                                     &::fdiml, data, size);

  SingleInputSingleOutputDiff<float>(&LIBC_NAMESPACE::sqrtf, &::sqrtf, data,
                                     size);
  SingleInputSingleOutputDiff<double>(&LIBC_NAMESPACE::sqrt, &::sqrt, data,
                                      size);
  SingleInputSingleOutputDiff<long double>(&LIBC_NAMESPACE::sqrtl, &::sqrtl,
                                           data, size);

  SingleInputSingleOutputWithSideEffectDiff<float, int>(&LIBC_NAMESPACE::frexpf,
                                                        &::frexpf, data, size);
  SingleInputSingleOutputWithSideEffectDiff<double, int>(&LIBC_NAMESPACE::frexp,
                                                         &::frexp, data, size);
  SingleInputSingleOutputWithSideEffectDiff<long double, int>(
      &LIBC_NAMESPACE::frexpl, &::frexpl, data, size);

  SingleInputSingleOutputWithSideEffectDiff<float, float>(
      &LIBC_NAMESPACE::modff, &::modff, data, size);
  SingleInputSingleOutputWithSideEffectDiff<double, double>(
      &LIBC_NAMESPACE::modf, &::modf, data, size);
  SingleInputSingleOutputWithSideEffectDiff<long double, long double>(
      &LIBC_NAMESPACE::modfl, &::modfl, data, size);

  TwoInputSingleOutputDiff<float, int>(&LIBC_NAMESPACE::ldexpf, &::ldexpf, data,
                                       size);
  TwoInputSingleOutputDiff<double, int>(&LIBC_NAMESPACE::ldexp, &::ldexp, data,
                                        size);
  TwoInputSingleOutputDiff<long double, int>(&LIBC_NAMESPACE::ldexpl, &::ldexpl,
                                             data, size);

  RemQuoDiff<float>(&LIBC_NAMESPACE::remquof, &::remquof, data, size);
  RemQuoDiff<double>(&LIBC_NAMESPACE::remquo, &::remquo, data, size);
  RemQuoDiff<long double>(&LIBC_NAMESPACE::remquol, &::remquol, data, size);

  return 0;
}
