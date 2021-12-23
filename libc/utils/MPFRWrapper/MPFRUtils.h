//===-- MPFRUtils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H
#define LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H

#include "src/__support/CPP/TypeTraits.h"
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
  Log,
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
  static constexpr bool VALUE = false;
};

template <typename T>
struct AreMatchingBinaryInputAndBinaryOutput<BinaryInput<T>, BinaryOutput<T>> {
  static constexpr bool VALUE = cpp::IsFloatingPointType<T>::Value;
};

template <typename T>
bool compare_unary_operation_single_output(Operation op, T input, T libc_output,
                                           double t);
template <typename T>
bool compare_unary_operation_two_outputs(Operation op, T input,
                                         const BinaryOutput<T> &libc_output,
                                         double t);
template <typename T>
bool compare_binary_operation_two_outputs(Operation op,
                                          const BinaryInput<T> &input,
                                          const BinaryOutput<T> &libc_output,
                                          double t);

template <typename T>
bool compare_binary_operation_one_output(Operation op,
                                         const BinaryInput<T> &input,
                                         T libc_output, double t);

template <typename T>
bool compare_ternary_operation_one_output(Operation op,
                                          const TernaryInput<T> &input,
                                          T libc_output, double t);

template <typename T>
void explain_unary_operation_single_output_error(Operation op, T input,
                                                 T match_value,
                                                 testutils::StreamWrapper &OS);
template <typename T>
void explain_unary_operation_two_outputs_error(
    Operation op, T input, const BinaryOutput<T> &match_value,
    testutils::StreamWrapper &OS);
template <typename T>
void explain_binary_operation_two_outputs_error(
    Operation op, const BinaryInput<T> &input,
    const BinaryOutput<T> &match_value, testutils::StreamWrapper &OS);

template <typename T>
void explain_binary_operation_one_output_error(Operation op,
                                               const BinaryInput<T> &input,
                                               T match_value,
                                               testutils::StreamWrapper &OS);

template <typename T>
void explain_ternary_operation_one_output_error(Operation op,
                                                const TernaryInput<T> &input,
                                                T match_value,
                                                testutils::StreamWrapper &OS);

template <Operation op, typename InputType, typename OutputType>
class MPFRMatcher : public testing::Matcher<OutputType> {
  InputType input;
  OutputType match_value;
  double ulp_tolerance;

public:
  MPFRMatcher(InputType testInput, double ulp_tolerance)
      : input(testInput), ulp_tolerance(ulp_tolerance) {}

  bool match(OutputType libcResult) {
    match_value = libcResult;
    return match(input, match_value, ulp_tolerance);
  }

  // This method is marked with NOLINT because it the name `explainError`
  // does not confirm to the coding style.
  void explainError(testutils::StreamWrapper &OS) override { // NOLINT
    explain_error(input, match_value, OS);
  }

private:
  template <typename T> static bool match(T in, T out, double tolerance) {
    return compare_unary_operation_single_output(op, in, out, tolerance);
  }

  template <typename T>
  static bool match(T in, const BinaryOutput<T> &out, double tolerance) {
    return compare_unary_operation_two_outputs(op, in, out, tolerance);
  }

  template <typename T>
  static bool match(const BinaryInput<T> &in, T out, double tolerance) {
    return compare_binary_operation_one_output(op, in, out, tolerance);
  }

  template <typename T>
  static bool match(BinaryInput<T> in, const BinaryOutput<T> &out,
                    double tolerance) {
    return compare_binary_operation_two_outputs(op, in, out, tolerance);
  }

  template <typename T>
  static bool match(const TernaryInput<T> &in, T out, double tolerance) {
    return compare_ternary_operation_one_output(op, in, out, tolerance);
  }

  template <typename T>
  static void explain_error(T in, T out, testutils::StreamWrapper &OS) {
    explain_unary_operation_single_output_error(op, in, out, OS);
  }

  template <typename T>
  static void explain_error(T in, const BinaryOutput<T> &out,
                            testutils::StreamWrapper &OS) {
    explain_unary_operation_two_outputs_error(op, in, out, OS);
  }

  template <typename T>
  static void explain_error(const BinaryInput<T> &in,
                            const BinaryOutput<T> &out,
                            testutils::StreamWrapper &OS) {
    explain_binary_operation_two_outputs_error(op, in, out, OS);
  }

  template <typename T>
  static void explain_error(const BinaryInput<T> &in, T out,
                            testutils::StreamWrapper &OS) {
    explain_binary_operation_one_output_error(op, in, out, OS);
  }

  template <typename T>
  static void explain_error(const TernaryInput<T> &in, T out,
                            testutils::StreamWrapper &OS) {
    explain_ternary_operation_one_output_error(op, in, out, OS);
  }
};

} // namespace internal

// Return true if the input and ouput types for the operation op are valid
// types.
template <Operation op, typename InputType, typename OutputType>
constexpr bool is_valid_operation() {
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
                                                          OutputType>::VALUE) ||
         (Operation::BeginTernaryOperationsSingleOuput < op &&
          op < Operation::EndTernaryOperationsSingleOutput &&
          cpp::IsFloatingPointType<OutputType>::Value &&
          cpp::IsSame<InputType, TernaryInput<OutputType>>::Value);
}

template <Operation op, typename InputType, typename OutputType>
__attribute__((no_sanitize("address")))
cpp::EnableIfType<is_valid_operation<op, InputType, OutputType>(),
                  internal::MPFRMatcher<op, InputType, OutputType>>
get_mpfr_matcher(InputType input, OutputType output_unused, double t) {
  return internal::MPFRMatcher<op, InputType, OutputType>(input, t);
}

enum class RoundingMode : uint8_t { Upward, Downward, TowardZero, Nearest };

template <typename T> T round(T x, RoundingMode mode);

template <typename T> bool round_to_long(T x, long &result);
template <typename T> bool round_to_long(T x, RoundingMode mode, long &result);

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc

#define EXPECT_MPFR_MATCH(op, input, match_value, tolerance)                   \
  EXPECT_THAT(match_value, __llvm_libc::testing::mpfr::get_mpfr_matcher<op>(   \
                               input, match_value, tolerance))

#define ASSERT_MPFR_MATCH(op, input, match_value, tolerance)                   \
  ASSERT_THAT(match_value, __llvm_libc::testing::mpfr::get_mpfr_matcher<op>(   \
                               input, match_value, tolerance))

#endif // LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H
