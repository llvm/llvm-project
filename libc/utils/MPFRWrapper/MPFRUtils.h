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

struct Tolerance {
  // Number of bits used to represent the fractional
  // part of a value of type 'float'.
  static constexpr unsigned int floatPrecision = 23;

  // Number of bits used to represent the fractional
  // part of a value of type 'double'.
  static constexpr unsigned int doublePrecision = 52;

  // The base precision of the number. For example, for values of
  // type float, the base precision is the value |floatPrecision|.
  unsigned int basePrecision;

  unsigned int width; // Number of valid LSB bits in |value|.

  // The bits in the tolerance value. The tolerance value will be
  // sum(bits[width - i] * 2 ^ (- basePrecision - i)) for |i| in
  // range [1, width].
  uint32_t bits;
};

enum Operation { OP_Abs, OP_Cos, OP_Sin, OP_Exp, OP_Exp2 };

namespace internal {

template <typename T>
bool compare(Operation op, T input, T libcOutput, const Tolerance &t);

template <typename T> class MPFRMatcher : public testing::Matcher<T> {
  static_assert(__llvm_libc::cpp::IsFloatingPointType<T>::Value,
                "MPFRMatcher can only be used with floating point values.");

  Operation operation;
  T input;
  Tolerance tolerance;
  T matchValue;

public:
  MPFRMatcher(Operation op, T testInput, Tolerance &t)
      : operation(op), input(testInput), tolerance(t) {}

  bool match(T libcResult) {
    matchValue = libcResult;
    return internal::compare(operation, input, libcResult, tolerance);
  }

  void explainError(testutils::StreamWrapper &OS) override;
};

} // namespace internal

template <typename T>
__attribute__((no_sanitize("address")))
internal::MPFRMatcher<T> getMPFRMatcher(Operation op, T input, Tolerance t) {
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
