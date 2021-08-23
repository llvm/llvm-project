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
  // Operations with take a single floating point number as input
  // and produce a single floating point number as output. The input
  // and output floating point numbers are of the same kind.
  BeginUnaryOperationsSingleOutput,
  Abs,
  Ceil,
  Cos,
  Exp,
  Exp2,
  Expm1,
  Floor,
  Mod2PI,
  ModPIOver2,
  ModPIOver4,
  Round,
  Sin,
  Sqrt,
  Tan,
  Trunc,
  EndUnaryOperationsSingleOutput,

  // Operations which take a single floating point nubmer as input
  // but produce two outputs. The first ouput is a floating point
  // number of the same type as the input. The second output is of type
  // 'int'.
  BeginUnaryOperationsTwoOutputs,
  Frexp, // Floating point output, the first output, is the fractional part.
  EndUnaryOperationsTwoOutputs,

  // Operations wich take two floating point nubmers of the same type as
  // input and produce a single floating point number of the same type as
  // output.
  BeginBinaryOperationsSingleOutput,
  Hypot,
  EndBinaryOperationsSingleOutput,

  // Operations which take two floating point numbers of the same type as
  // input and produce two outputs. The first output is a floating nubmer of
  // the same type as the inputs. The second output is af type 'int'.
  BeginBinaryOperationsTwoOutputs,
  RemQuo, // The first output, the floating point output, is the remainder.
  EndBinaryOperationsTwoOutputs,

  // Operations which take three floating point nubmers of the same type as
  // input and produce a single floating point number of the same type as
  // output.
  BeginTernaryOperationsSingleOuput,
  Fma,
  EndTernaryOperationsSingleOutput,
};

template <typename T> struct BinaryInput {
  static_assert(
      __llvm_libc::cpp::IsFloatingPointType<T>::Value,
      "Template parameter of BinaryInput must be a floating point type.");

  using Type = T;
  T x, y;
};

template <typename T> struct TernaryInput {
  static_assert(
      __llvm_libc::cpp::IsFloatingPointType<T>::Value,
      "Template parameter of TernaryInput must be a floating point type.");

  using Type = T;
  T x, y, z;
};

template <typename T> struct BinaryOutput {
  T f;
  int i;
};

namespace internal {

template <typename T1, typename T2>
struct AreMatchingBinaryInputAndBinaryOutput {
  static constexpr bool value = false;
};

template <typename T>
struct AreMatchingBinaryInputAndBinaryOutput<BinaryInput<T>, BinaryOutput<T>> {
  static constexpr bool value = cpp::IsFloatingPointType<T>::Value;
};

template <typename T>
bool compareUnaryOperationSingleOutput(Operation op, T input, T libcOutput,
                                       double t);
template <typename T>
bool compareUnaryOperationTwoOutputs(Operation op, T input,
                                     const BinaryOutput<T> &libcOutput,
                                     double t);
template <typename T>
bool compareBinaryOperationTwoOutputs(Operation op, const BinaryInput<T> &input,
                                      const BinaryOutput<T> &libcOutput,
                                      double t);

template <typename T>
bool compareBinaryOperationOneOutput(Operation op, const BinaryInput<T> &input,
                                     T libcOutput, double t);

template <typename T>
bool compareTernaryOperationOneOutput(Operation op,
                                      const TernaryInput<T> &input,
                                      T libcOutput, double t);

template <typename T>
void explainUnaryOperationSingleOutputError(Operation op, T input, T matchValue,
                                            testutils::StreamWrapper &OS);
template <typename T>
void explainUnaryOperationTwoOutputsError(Operation op, T input,
                                          const BinaryOutput<T> &matchValue,
                                          testutils::StreamWrapper &OS);
template <typename T>
void explainBinaryOperationTwoOutputsError(Operation op,
                                           const BinaryInput<T> &input,
                                           const BinaryOutput<T> &matchValue,
                                           testutils::StreamWrapper &OS);

template <typename T>
void explainBinaryOperationOneOutputError(Operation op,
                                          const BinaryInput<T> &input,
                                          T matchValue,
                                          testutils::StreamWrapper &OS);

template <typename T>
void explainTernaryOperationOneOutputError(Operation op,
                                           const TernaryInput<T> &input,
                                           T matchValue,
                                           testutils::StreamWrapper &OS);

template <Operation op, typename InputType, typename OutputType>
class MPFRMatcher : public testing::Matcher<OutputType> {
  InputType input;
  OutputType matchValue;
  double ulpTolerance;

public:
  MPFRMatcher(InputType testInput, double ulpTolerance)
      : input(testInput), ulpTolerance(ulpTolerance) {}

  bool match(OutputType libcResult) {
    matchValue = libcResult;
    return match(input, matchValue, ulpTolerance);
  }

  void explainError(testutils::StreamWrapper &OS) override {
    explainError(input, matchValue, OS);
  }

private:
  template <typename T> static bool match(T in, T out, double tolerance) {
    return compareUnaryOperationSingleOutput(op, in, out, tolerance);
  }

  template <typename T>
  static bool match(T in, const BinaryOutput<T> &out, double tolerance) {
    return compareUnaryOperationTwoOutputs(op, in, out, tolerance);
  }

  template <typename T>
  static bool match(const BinaryInput<T> &in, T out, double tolerance) {
    return compareBinaryOperationOneOutput(op, in, out, tolerance);
  }

  template <typename T>
  static bool match(BinaryInput<T> in, const BinaryOutput<T> &out,
                    double tolerance) {
    return compareBinaryOperationTwoOutputs(op, in, out, tolerance);
  }

  template <typename T>
  static bool match(const TernaryInput<T> &in, T out, double tolerance) {
    return compareTernaryOperationOneOutput(op, in, out, tolerance);
  }

  template <typename T>
  static void explainError(T in, T out, testutils::StreamWrapper &OS) {
    explainUnaryOperationSingleOutputError(op, in, out, OS);
  }

  template <typename T>
  static void explainError(T in, const BinaryOutput<T> &out,
                           testutils::StreamWrapper &OS) {
    explainUnaryOperationTwoOutputsError(op, in, out, OS);
  }

  template <typename T>
  static void explainError(const BinaryInput<T> &in, const BinaryOutput<T> &out,
                           testutils::StreamWrapper &OS) {
    explainBinaryOperationTwoOutputsError(op, in, out, OS);
  }

  template <typename T>
  static void explainError(const BinaryInput<T> &in, T out,
                           testutils::StreamWrapper &OS) {
    explainBinaryOperationOneOutputError(op, in, out, OS);
  }

  template <typename T>
  static void explainError(const TernaryInput<T> &in, T out,
                           testutils::StreamWrapper &OS) {
    explainTernaryOperationOneOutputError(op, in, out, OS);
  }
};

} // namespace internal

// Return true if the input and ouput types for the operation op are valid
// types.
template <Operation op, typename InputType, typename OutputType>
constexpr bool isValidOperation() {
  return (Operation::BeginUnaryOperationsSingleOutput < op &&
          op < Operation::EndUnaryOperationsSingleOutput &&
          cpp::IsSame<InputType, OutputType>::Value &&
          cpp::IsFloatingPointType<InputType>::Value) ||
         (Operation::BeginUnaryOperationsTwoOutputs < op &&
          op < Operation::EndUnaryOperationsTwoOutputs &&
          cpp::IsFloatingPointType<InputType>::Value &&
          cpp::IsSame<OutputType, BinaryOutput<InputType>>::Value) ||
         (Operation::BeginBinaryOperationsSingleOutput < op &&
          op < Operation::EndBinaryOperationsSingleOutput &&
          cpp::IsFloatingPointType<OutputType>::Value &&
          cpp::IsSame<InputType, BinaryInput<OutputType>>::Value) ||
         (Operation::BeginBinaryOperationsTwoOutputs < op &&
          op < Operation::EndBinaryOperationsTwoOutputs &&
          internal::AreMatchingBinaryInputAndBinaryOutput<InputType,
                                                          OutputType>::value) ||
         (Operation::BeginTernaryOperationsSingleOuput < op &&
          op < Operation::EndTernaryOperationsSingleOutput &&
          cpp::IsFloatingPointType<OutputType>::Value &&
          cpp::IsSame<InputType, TernaryInput<OutputType>>::Value);
}

template <Operation op, typename InputType, typename OutputType>
__attribute__((no_sanitize("address")))
cpp::EnableIfType<isValidOperation<op, InputType, OutputType>(),
                  internal::MPFRMatcher<op, InputType, OutputType>>
getMPFRMatcher(InputType input, OutputType outputUnused, double t) {
  return internal::MPFRMatcher<op, InputType, OutputType>(input, t);
}

enum class RoundingMode : uint8_t { Upward, Downward, TowardZero, Nearest };

template <typename T> T Round(T x, RoundingMode mode);

template <typename T> bool RoundToLong(T x, long &result);
template <typename T> bool RoundToLong(T x, RoundingMode mode, long &result);

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc

#define EXPECT_MPFR_MATCH(op, input, matchValue, tolerance)                    \
  EXPECT_THAT(matchValue, __llvm_libc::testing::mpfr::getMPFRMatcher<op>(      \
                              input, matchValue, tolerance))

#define ASSERT_MPFR_MATCH(op, input, matchValue, tolerance)                    \
  ASSERT_THAT(matchValue, __llvm_libc::testing::mpfr::getMPFRMatcher<op>(      \
                              input, matchValue, tolerance))

#endif // LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H
