//===-- MPCUtils.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_MPCWRAPPER_MPCUTILS_H
#define LLVM_LIBC_UTILS_MPCWRAPPER_MPCUTILS_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/type_traits/is_complex.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/complex_types.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace testing {
namespace mpc {

enum class Operation {
  // Operations which take a single complex floating point number as input
  // and produce a single floating point number as output which has the same
  // floating point type as the real/imaginary part of the input.
  BeginUnaryOperationsSingleOutputDifferentOutputType,
  Carg,
  Cabs,
  EndUnaryOperationsSingleOutputDifferentOutputType,

  // Operations which take a single complex floating point number as input
  // and produce a single complex floating point number of the same kind
  // as output.
  BeginUnaryOperationsSingleOutputSameOutputType,
  Csqrt,
  Clog,
  Cexp,
  Csinh,
  Ccosh,
  Ctanh,
  Casinh,
  Cacosh,
  Catanh,
  Csin,
  Ccos,
  Ctan,
  Casin,
  Cacos,
  Catan,
  EndUnaryOperationsSingleOutputSameOutputType,

  // Operations which take two complex floating point numbers as input
  // and produce a single complex floating point number of the same kind
  // as output.
  BeginBinaryOperationsSingleOutput,
  Cpow,
  EndBinaryOperationsSingleOutput,
};

using LIBC_NAMESPACE::fputil::testing::RoundingMode;

struct MPCRoundingMode {
  RoundingMode Rrnd;
  RoundingMode Irnd;
};

template <typename T> struct BinaryInput {
  static_assert(LIBC_NAMESPACE::cpp::is_complex_v<T>,
                "Template parameter of BinaryInput must be a complex floating "
                "point type.");

  using Type = T;
  T x, y;
};

namespace internal {

template <typename InputType, typename OutputType>
bool compare_unary_operation_single_output(Operation op, InputType input,
                                           OutputType libc_output,
                                           double ulp_tolerance,
                                           MPCRoundingMode rounding);

template <typename InputType, typename OutputType>
bool compare_binary_operation_one_output(Operation op,
                                         const BinaryInput<InputType> &input,
                                         OutputType libc_output,
                                         double ulp_tolerance,
                                         MPCRoundingMode rounding);

template <typename InputType, typename OutputType>
void explain_unary_operation_single_output_error(Operation op, InputType input,
                                                 OutputType match_value,
                                                 double ulp_tolerance,
                                                 MPCRoundingMode rounding);

template <typename InputType, typename OutputType>
void explain_binary_operation_one_output_error(
    Operation op, const BinaryInput<InputType> &input, OutputType match_value,
    double ulp_tolerance, MPCRoundingMode rounding);

template <Operation op, typename InputType, typename OutputType>
class MPCMatcher : public testing::Matcher<OutputType> {
private:
  InputType input;
  OutputType match_value;
  double ulp_tolerance;
  MPCRoundingMode rounding;

public:
  MPCMatcher(InputType testInput, double ulp_tolerance,
             MPCRoundingMode rounding)
      : input(testInput), ulp_tolerance(ulp_tolerance), rounding(rounding) {}

  bool match(OutputType libcResult) {
    match_value = libcResult;
    return match(input, match_value);
  }

private:
  template <typename InType, typename OutType>
  bool match(InType in, OutType out) {
    if (cpp::is_same_v<InType, OutType>) {
      return compare_unary_operation_single_output_same_type(op, in, out, ulp_tolerance, rounding);  
    } else {
      return compare_unary_operation_single_output_different_type(op, in, out, ulp_tolerance, rounding);
    }
  }

  template <typename T, typename U>
  bool match(const BinaryInput<T> &in, U out) {
    return compare_binary_operation_one_output(op, in, out, ulp_tolerance,
                                               rounding);
  }

  template <typename InType, typename OutType>
  void explain_error(InType in, OutType out) {
    if (cpp::is_same_v<InType, OutType>) {
      explain_unary_operation_single_output_same_type_error(op, in, out, ulp_tolerance, rounding);
    } else {
      explain_unary_operation_single_output_different_type_error(op, in, out, ulp_tolerance, rounding);
    }
  }

  template <typename T, typename U>
  void explain_error(const BinaryInput<T> &in, U out) {
    explain_binary_operation_one_output_error(op, in, out, ulp_tolerance,
                                              rounding);
  }
};

} // namespace internal

template <typename T> struct get_real;

template <> struct get_real<_Complex float> {
  using type = float;
};
template <> struct get_real<_Complex double> {
  using type = double;
};
template <> struct get_real<_Complex long double> {
  using type = long double;
};

#if defined(LIBC_TYPES_HAS_CFLOAT16)
template <> struct get_real<cfloat16> {
  using type = float16;
};
#endif
#ifdef LIBC_TYPES_CFLOAT128_IS_NOT_COMPLEX_LONG_DOUBLE
template <> struct get_real<cfloat128> {
  using type = float128;
};
#endif

template <typename T> using get_real_t = typename get_real<T>::type;

// Return true if the input and ouput types for the operation op are valid
// types.
template <Operation op, typename InputType, typename OutputType>
constexpr bool is_valid_operation() {
  return (Operation::BeginBinaryOperationsSingleOutput < op &&
          op < Operation::EndBinaryOperationsSingleOutput &&
          cpp::is_complex_type_same<InputType, OutputType> &&
          cpp::is_complex_v<InputType>) ||
         (Operation::BeginUnaryOperationsSingleOutputSameOutputType < op &&
          op < Operation::EndUnaryOperationsSingleOutputSameOutputType &&
          cpp::is_complex_type_same<InputType, OutputType> &&
          cpp::is_complex_v<InputType>) ||
         (Operation::BeginUnaryOperationsSingleOutputDifferentOutputType < op &&
          op < Operation::EndUnaryOperationsSingleOutputDifferentOutputType &&
          cpp::is_same_v<get_real_t<InputType>, OutputType> &&
          cpp::is_complex_v<InputType>);
}

template <Operation op, typename InputType, typename OutputType>
cpp::enable_if_t<is_valid_operation<op, InputType, OutputType>(),
                 internal::MPCMatcher<op, InputType, OutputType>>
get_mpc_matcher(InputType input, [[maybe_unused]] OutputType output,
                double ulp_tolerance, MPCRoundingMode rounding) {
  return internal::MPCMatcher<op, InputType, OutputType>(input, ulp_tolerance,
                                                         rounding);
}

} // namespace mpc
} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#define EXPECT_MPC_MATCH_DEFAULT(op, input, match_value, ulp_tolerance)        \
  EXPECT_THAT(                                                                 \
      match_value,                                                             \
      LIBC_NAMESPACE::testing::mpc::get_mpc_matcher<op>(                       \
          input, match_value, ulp_tolerance,                                   \
          MPCRoundingMode{                                                     \
              LIBC_NAMESPACE::fputil::testing::RoundingMode::Nearest,          \
              LIBC_NAMESPACE::fputil::testing::RoundingMode::Nearest}))

#define EXPECT_MPC_MATCH_ROUNDING(op, input, match_value, ulp_tolerance,       \
                                  Rrounding, Irounding)                        \
  EXPECT_THAT(match_value, LIBC_NAMESPACE::testing::mpc::get_mpc_matcher<op>(  \
                               input, match_value, ulp_tolerance,              \
                               MPCRoundingMode{Rrounding, Irounding}))

#define EXPECT_MPC_MATCH_ALL_ROUNDING_HELPER(                                  \
    i, j, op, input, match_value, ulp_tolerance, Rrounding, Irounding)         \
  {                                                                            \
    MPCRND::ForceRoundingMode __r##i##j(Rrounding);                            \
    MPCRND::ForceRoundingMode __i##i##j(Irounding);                            \
    if (__r##i##j.success && __i##i##j.success) {                              \
      EXPECT_MPC_MATCH_ROUNDING(                                               \
          match_value,                                                         \
          LIBC_NAMESPACE::testing::mpc::get_mpc_matcher<op>(                   \
              input, match_value, ulp_tolerance, Rrounding, Irounding))        \
    }                                                                          \
  }

#define EXPECT_MPC_MATCH_ALL_ROUNDING(op, input, match_value, ulp_tolerance)   \
  {                                                                            \
    namespace MPCRND = LIBC_NAMESPACE::fputil::testing;                        \
    for (int i = 0; i < 5; i++) {                                              \
      for (int j = 0; j < 5; j++) {                                            \
        RoundingMode r_mode = static_cast<RoundingMode>(i);                    \
        RoundingMode i_mode = static_cast<RoundingMode>(j);                    \
        EXPECT_MPC_MATCH_ALL_ROUNDING_HELPER(i, j, op, input, match_value,     \
                                             ulp_tolerance, r_mode, i_mode);   \
      }                                                                        \
    }                                                                          \
  }

#define TEST_MPC_MATCH_ROUNDING(op, input, match_value, ulp_tolerance,         \
                                Rrounding, Irounding)                          \
  LIBC_NAMESPACE::testing::mpc::get_mpc_matcher<op>(                           \
      input, match_value, ulp_tolerance,                                       \
      MPCRoundingMode{Rrounding, Irounding})                                   \
      .match(match_value)

#define ASSERT_MPC_MATCH_DEFAULT(op, input, match_value, ulp_tolerance)        \
  ASSERT_THAT(                                                                 \
      match_value,                                                             \
      LIBC_NAMESPACE::testing::mpc::get_mpc_matcher<op>(                       \
          input, match_value, ulp_tolerance,                                   \
          MPCRoundingMode{                                                     \
              LIBC_NAMESPACE::fputil::testing::RoundingMode::Nearest,          \
              LIBC_NAMESPACE::fputil::testing::RoundingMode::Nearest}))

#define ASSERT_MPC_MATCH_ROUNDING(op, input, match_value, ulp_tolerance,       \
                                  Rrounding, Irounding)                        \
  ASSERT_THAT(match_value, LIBC_NAMESPACE::testing::mpc::get_mpc_matcher<op>(  \
                               input, match_value, ulp_tolerance,              \
                               MPCRoundingMode{Rrounding, Irounding}))

#define ASSERT_MPC_MATCH_ALL_ROUNDING_HELPER(                                  \
    i, j, op, input, match_value, ulp_tolerance, Rrounding, Irounding)         \
  {                                                                            \
    MPCRND::ForceRoundingMode __r##i##j(Rrounding);                            \
    MPCRND::ForceRoundingMode __i##i##j(Irounding);                            \
    if (__r##i##j.success && __i##i##j.success) {                              \
      ASSERT_MPC_MATCH_ROUNDING(                                               \
          match_value,                                                         \
          LIBC_NAMESPACE::testing::mpc::get_mpc_matcher<op>(                   \
              input, match_value, ulp_tolerance, Rrounding, Irounding))        \
    }                                                                          \
  }

#define ASSERT_MPC_MATCH_ALL_ROUNDING(op, input, match_value, ulp_tolerance)   \
  {                                                                            \
    namespace MPCRND = LIBC_NAMESPACE::fputil::testing;                        \
    for (int i = 0; i < 5; i++) {                                              \
      for (int j = 0; j < 5; j++) {                                            \
        RoundingMode r_mode = static_cast<RoundingMode>(i);                    \
        RoundingMode i_mode = static_cast<RoundingMode>(j);                    \
        ASSERT_MPC_MATCH_ALL_ROUNDING_HELPER(i, j, op, input, match_value,     \
                                             ulp_tolerance, r_mode, i_mode);   \
      }                                                                        \
    }                                                                          \
  }

#endif // LLVM_LIBC_UTILS_MPCWRAPPER_MPCUTILS_H
