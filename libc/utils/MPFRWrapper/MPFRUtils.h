//===-- MPFRUtils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H
#define LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H

#include "utils/CPP/TypeTraits.h"
#include "utils/UnitTest/Test.h"

#include <stdint.h>

namespace __llvm_libc {
namespace testing {
namespace mpfr {

enum class Operation : int {
  Abs,
  Ceil,
  Cos,
  Exp,
  Exp2,
  Floor,
  Round,
  Sin,
  Sqrt,
  Trunc
};

namespace internal {

template <typename T>
bool compare(Operation op, T input, T libcOutput, double t);

template <typename T> class MPFRMatcher : public testing::Matcher<T> {
  static_assert(__llvm_libc::cpp::IsFloatingPointType<T>::Value,
                "MPFRMatcher can only be used with floating point values.");

  Operation operation;
  T input;
  T matchValue;
  double ulpTolerance;

public:
  MPFRMatcher(Operation op, T testInput, double ulpTolerance)
      : operation(op), input(testInput), ulpTolerance(ulpTolerance) {}

  bool match(T libcResult) {
    matchValue = libcResult;
    return internal::compare(operation, input, libcResult, ulpTolerance);
  }

  void explainError(testutils::StreamWrapper &OS) override;
};

} // namespace internal

template <typename T, typename U>
__attribute__((no_sanitize("address")))
typename cpp::EnableIfType<cpp::IsSameV<U, double>, internal::MPFRMatcher<T>>
getMPFRMatcher(Operation op, T input, U t) {
  static_assert(
      __llvm_libc::cpp::IsFloatingPointType<T>::Value,
      "getMPFRMatcher can only be used to match floating point results.");
  return internal::MPFRMatcher<T>(op, input, t);
}

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc

#define EXPECT_MPFR_MATCH(op, input, matchValue, tolerance)                    \
  EXPECT_THAT(matchValue, __llvm_libc::testing::mpfr::getMPFRMatcher(          \
                              op, input, tolerance))

#define ASSERT_MPFR_MATCH(op, input, matchValue, tolerance)                    \
  ASSERT_THAT(matchValue, __llvm_libc::testing::mpfr::getMPFRMatcher(          \
                              op, input, tolerance))

#endif // LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H
