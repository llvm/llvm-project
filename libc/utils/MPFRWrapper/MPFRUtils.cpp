//===-- Utils which wrap MPFR ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPFRUtils.h"
#include "MPCommon.h"

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/fpbits_str.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {
namespace mpfr {
namespace internal {

template <typename InputType>
cpp::enable_if_t<cpp::is_floating_point_v<InputType>, MPFRNumber>
unary_operation(Operation op, InputType input, unsigned int precision,
                RoundingMode rounding) {
  MPFRNumber mpfrInput(input, precision, rounding);
  switch (op) {
  case Operation::Abs:
    return mpfrInput.abs();
  case Operation::Acos:
    return mpfrInput.acos();
  case Operation::Acosh:
    return mpfrInput.acosh();
  case Operation::Acospi:
    return mpfrInput.acospi();
  case Operation::Asin:
    return mpfrInput.asin();
  case Operation::Asinh:
    return mpfrInput.asinh();
  case Operation::Asinpi:
    return mpfrInput.asinpi();
  case Operation::Atan:
    return mpfrInput.atan();
  case Operation::Atanh:
    return mpfrInput.atanh();
  case Operation::Atanpi:
    return mpfrInput.atanpi();
  case Operation::Cbrt:
    return mpfrInput.cbrt();
  case Operation::Ceil:
    return mpfrInput.ceil();
  case Operation::Cos:
    return mpfrInput.cos();
  case Operation::Cosh:
    return mpfrInput.cosh();
  case Operation::Cospi:
    return mpfrInput.cospi();
  case Operation::Erf:
    return mpfrInput.erf();
  case Operation::Exp:
    return mpfrInput.exp();
  case Operation::Exp2:
    return mpfrInput.exp2();
  case Operation::Exp2m1:
    return mpfrInput.exp2m1();
  case Operation::Exp10:
    return mpfrInput.exp10();
  case Operation::Exp10m1:
    return mpfrInput.exp10m1();
  case Operation::Expm1:
    return mpfrInput.expm1();
  case Operation::Floor:
    return mpfrInput.floor();
  case Operation::Log:
    return mpfrInput.log();
  case Operation::Log2:
    return mpfrInput.log2();
  case Operation::Log10:
    return mpfrInput.log10();
  case Operation::Log1p:
    return mpfrInput.log1p();
  case Operation::Mod2PI:
    return mpfrInput.mod_2pi();
  case Operation::ModPIOver2:
    return mpfrInput.mod_pi_over_2();
  case Operation::ModPIOver4:
    return mpfrInput.mod_pi_over_4();
  case Operation::Round:
    return mpfrInput.round();
  case Operation::RoundEven:
    return mpfrInput.roundeven();
  case Operation::Sin:
    return mpfrInput.sin();
  case Operation::Sinpi:
    return mpfrInput.sinpi();
  case Operation::Sinh:
    return mpfrInput.sinh();
  case Operation::Sqrt:
    return mpfrInput.sqrt();
  case Operation::Tan:
    return mpfrInput.tan();
  case Operation::Tanh:
    return mpfrInput.tanh();
  case Operation::Tanpi:
    return mpfrInput.tanpi();
  case Operation::Trunc:
    return mpfrInput.trunc();
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::enable_if_t<cpp::is_floating_point_v<InputType>, MPFRNumber>
unary_operation_two_outputs(Operation op, InputType input, int &output,
                            unsigned int precision, RoundingMode rounding) {
  MPFRNumber mpfrInput(input, precision, rounding);
  switch (op) {
  case Operation::Frexp:
    return mpfrInput.frexp(output);
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::enable_if_t<cpp::is_floating_point_v<InputType>, MPFRNumber>
binary_operation_one_output(Operation op, InputType x, InputType y,
                            unsigned int precision, RoundingMode rounding) {
  MPFRNumber inputX(x, precision, rounding);
  MPFRNumber inputY(y, precision, rounding);
  switch (op) {
  case Operation::Add:
    return inputX.add(inputY);
  case Operation::Atan2:
    return inputX.atan2(inputY);
  case Operation::Div:
    return inputX.div(inputY);
  case Operation::Fmod:
    return inputX.fmod(inputY);
  case Operation::Hypot:
    return inputX.hypot(inputY);
  case Operation::Mul:
    return inputX.mul(inputY);
  case Operation::Pow:
    return inputX.pow(inputY);
  case Operation::Sub:
    return inputX.sub(inputY);
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::enable_if_t<cpp::is_floating_point_v<InputType>, MPFRNumber>
binary_operation_two_outputs(Operation op, InputType x, InputType y,
                             int &output, unsigned int precision,
                             RoundingMode rounding) {
  MPFRNumber inputX(x, precision, rounding);
  MPFRNumber inputY(y, precision, rounding);
  switch (op) {
  case Operation::RemQuo:
    return inputX.remquo(inputY, output);
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::enable_if_t<cpp::is_floating_point_v<InputType>, MPFRNumber>
ternary_operation_one_output(Operation op, InputType x, InputType y,
                             InputType z, unsigned int precision,
                             RoundingMode rounding) {
  // For FMA function, we just need to compare with the mpfr_fma with the same
  // precision as InputType.  Using higher precision as the intermediate results
  // to compare might incorrectly fail due to double-rounding errors.
  MPFRNumber inputX(x, precision, rounding);
  MPFRNumber inputY(y, precision, rounding);
  MPFRNumber inputZ(z, precision, rounding);
  switch (op) {
  case Operation::Fma:
    return inputX.fma(inputY, inputZ);
  default:
    __builtin_unreachable();
  }
}

// Remark: For all the explain_*_error functions, we will use std::stringstream
// to build the complete error messages before sending it to the outstream `OS`
// once at the end.  This will stop the error messages from interleaving when
// the tests are running concurrently.
template <typename InputType, typename OutputType>
void explain_unary_operation_single_output_error(Operation op, InputType input,
                                                 OutputType matchValue,
                                                 double ulp_tolerance,
                                                 RoundingMode rounding) {
  unsigned int precision = get_precision<InputType>(ulp_tolerance);
  MPFRNumber mpfrInput(input, precision);
  MPFRNumber mpfr_result;
  mpfr_result = unary_operation(op, input, precision, rounding);
  MPFRNumber mpfrMatchValue(matchValue);
  cpp::array<char, 1024> msg_buf;
  cpp::StringStream msg(msg_buf);
  msg << "Match value not within tolerance value of MPFR result:\n"
      << "  Input decimal: " << mpfrInput.str() << '\n';
  msg << "     Input bits: " << str(FPBits<InputType>(input)) << '\n';
  msg << '\n' << "  Match decimal: " << mpfrMatchValue.str() << '\n';
  msg << "     Match bits: " << str(FPBits<OutputType>(matchValue)) << '\n';
  msg << '\n' << "    MPFR result: " << mpfr_result.str() << '\n';
  msg << "   MPFR rounded: "
      << str(FPBits<OutputType>(mpfr_result.as<OutputType>())) << '\n';
  msg << '\n';
  msg << "      ULP error: " << mpfr_result.ulp_as_mpfr_number(matchValue).str()
      << '\n';
  if (msg.overflow())
    __builtin_unreachable();
  tlog << msg.str();
}

template void explain_unary_operation_single_output_error(Operation op, float,
                                                          float, double,
                                                          RoundingMode);
template void explain_unary_operation_single_output_error(Operation op, double,
                                                          double, double,
                                                          RoundingMode);
template void explain_unary_operation_single_output_error(Operation op,
                                                          long double,
                                                          long double, double,
                                                          RoundingMode);
template void explain_unary_operation_single_output_error(Operation op, double,
                                                          float, double,
                                                          RoundingMode);
template void explain_unary_operation_single_output_error(Operation op,
                                                          long double, float,
                                                          double, RoundingMode);
template void explain_unary_operation_single_output_error(Operation op,
                                                          long double, double,
                                                          double, RoundingMode);

#ifdef LIBC_TYPES_HAS_FLOAT16
template void explain_unary_operation_single_output_error(Operation op, float16,
                                                          float16, double,
                                                          RoundingMode);
template void explain_unary_operation_single_output_error(Operation op, float,
                                                          float16, double,
                                                          RoundingMode);
template void explain_unary_operation_single_output_error(Operation op, double,
                                                          float16, double,
                                                          RoundingMode);
template void explain_unary_operation_single_output_error(Operation op,
                                                          long double, float16,
                                                          double, RoundingMode);
#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
template void explain_unary_operation_single_output_error(Operation op,
                                                          float128, float16,
                                                          double, RoundingMode);
#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
template void explain_unary_operation_single_output_error(Operation op,
                                                          float128, float128,
                                                          double, RoundingMode);
template void explain_unary_operation_single_output_error(Operation op,
                                                          float128, float,
                                                          double, RoundingMode);
template void explain_unary_operation_single_output_error(Operation op,
                                                          float128, double,
                                                          double, RoundingMode);
template void explain_unary_operation_single_output_error(Operation op,
                                                          float128, long double,
                                                          double, RoundingMode);
#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE

template <typename T>
void explain_unary_operation_two_outputs_error(
    Operation op, T input, const BinaryOutput<T> &libc_result,
    double ulp_tolerance, RoundingMode rounding) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfrInput(input, precision);
  int mpfrIntResult;
  MPFRNumber mpfr_result = unary_operation_two_outputs(op, input, mpfrIntResult,
                                                       precision, rounding);

  if (mpfrIntResult != libc_result.i) {
    tlog << "MPFR integral result: " << mpfrIntResult << '\n'
         << "Libc integral result: " << libc_result.i << '\n';
  } else {
    tlog << "Integral result from libc matches integral result from MPFR.\n";
  }

  MPFRNumber mpfrMatchValue(libc_result.f);
  tlog
      << "Libc floating point result is not within tolerance value of the MPFR "
      << "result.\n\n";

  tlog << "            Input decimal: " << mpfrInput.str() << "\n\n";

  tlog << "Libc floating point value: " << mpfrMatchValue.str() << '\n';
  tlog << " Libc floating point bits: " << str(FPBits<T>(libc_result.f))
       << '\n';
  tlog << "\n\n";

  tlog << "              MPFR result: " << mpfr_result.str() << '\n';
  tlog << "             MPFR rounded: " << str(FPBits<T>(mpfr_result.as<T>()))
       << '\n';
  tlog << '\n'
       << "                ULP error: "
       << mpfr_result.ulp_as_mpfr_number(libc_result.f).str() << '\n';
}

template void explain_unary_operation_two_outputs_error<float>(
    Operation, float, const BinaryOutput<float> &, double, RoundingMode);
template void explain_unary_operation_two_outputs_error<double>(
    Operation, double, const BinaryOutput<double> &, double, RoundingMode);
template void explain_unary_operation_two_outputs_error<long double>(
    Operation, long double, const BinaryOutput<long double> &, double,
    RoundingMode);

template <typename T>
void explain_binary_operation_two_outputs_error(
    Operation op, const BinaryInput<T> &input,
    const BinaryOutput<T> &libc_result, double ulp_tolerance,
    RoundingMode rounding) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfrX(input.x, precision);
  MPFRNumber mpfrY(input.y, precision);
  int mpfrIntResult;
  MPFRNumber mpfr_result = binary_operation_two_outputs(
      op, input.x, input.y, mpfrIntResult, precision, rounding);
  MPFRNumber mpfrMatchValue(libc_result.f);

  tlog << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str() << '\n'
       << "MPFR integral result: " << mpfrIntResult << '\n'
       << "Libc integral result: " << libc_result.i << '\n'
       << "Libc floating point result: " << mpfrMatchValue.str() << '\n'
       << "               MPFR result: " << mpfr_result.str() << '\n';
  tlog << "Libc floating point result bits: " << str(FPBits<T>(libc_result.f))
       << '\n';
  tlog << "              MPFR rounded bits: "
       << str(FPBits<T>(mpfr_result.as<T>())) << '\n';
  tlog << "ULP error: " << mpfr_result.ulp_as_mpfr_number(libc_result.f).str()
       << '\n';
}

template void explain_binary_operation_two_outputs_error<float>(
    Operation, const BinaryInput<float> &, const BinaryOutput<float> &, double,
    RoundingMode);
template void explain_binary_operation_two_outputs_error<double>(
    Operation, const BinaryInput<double> &, const BinaryOutput<double> &,
    double, RoundingMode);
template void explain_binary_operation_two_outputs_error<long double>(
    Operation, const BinaryInput<long double> &,
    const BinaryOutput<long double> &, double, RoundingMode);

template <typename InputType, typename OutputType>
void explain_binary_operation_one_output_error(
    Operation op, const BinaryInput<InputType> &input, OutputType libc_result,
    double ulp_tolerance, RoundingMode rounding) {
  unsigned int precision = get_precision<InputType>(ulp_tolerance);
  MPFRNumber mpfrX(input.x, precision);
  MPFRNumber mpfrY(input.y, precision);
  FPBits<InputType> xbits(input.x);
  FPBits<InputType> ybits(input.y);
  MPFRNumber mpfr_result =
      binary_operation_one_output(op, input.x, input.y, precision, rounding);
  MPFRNumber mpfrMatchValue(libc_result);

  tlog << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str() << '\n';
  tlog << "First input bits: " << str(FPBits<InputType>(input.x)) << '\n';
  tlog << "Second input bits: " << str(FPBits<InputType>(input.y)) << '\n';

  tlog << "Libc result: " << mpfrMatchValue.str() << '\n'
       << "MPFR result: " << mpfr_result.str() << '\n';
  tlog << "Libc floating point result bits: "
       << str(FPBits<OutputType>(libc_result)) << '\n';
  tlog << "              MPFR rounded bits: "
       << str(FPBits<OutputType>(mpfr_result.as<OutputType>())) << '\n';
  tlog << "ULP error: " << mpfr_result.ulp_as_mpfr_number(libc_result).str()
       << '\n';
}

template void
explain_binary_operation_one_output_error(Operation, const BinaryInput<float> &,
                                          float, double, RoundingMode);
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<double> &, float, double, RoundingMode);
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<double> &, double, double, RoundingMode);
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<long double> &, float, double, RoundingMode);
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<long double> &, double, double, RoundingMode);
template void
explain_binary_operation_one_output_error(Operation,
                                          const BinaryInput<long double> &,
                                          long double, double, RoundingMode);
#ifdef LIBC_TYPES_HAS_FLOAT16
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<float16> &, float16, double, RoundingMode);
template void
explain_binary_operation_one_output_error(Operation, const BinaryInput<float> &,
                                          float16, double, RoundingMode);
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<double> &, float16, double, RoundingMode);
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<long double> &, float16, double, RoundingMode);
#endif
#if defined(LIBC_TYPES_HAS_FLOAT128) &&                                        \
    defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<float128> &, float128, double, RoundingMode);
#endif
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<bfloat16> &, bfloat16, double, RoundingMode);
template void
explain_binary_operation_one_output_error(Operation, const BinaryInput<float> &,
                                          bfloat16, double, RoundingMode);
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<double> &, bfloat16, double, RoundingMode);
template void
explain_binary_operation_one_output_error(Operation,
                                          const BinaryInput<long double> &,
                                          bfloat16, double, RoundingMode);
#if defined(LIBC_TYPES_HAS_FLOAT128) &&                                        \
    defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)
template void explain_binary_operation_one_output_error(
    Operation, const BinaryInput<float128> &, bfloat16, double, RoundingMode);
#endif // defined(LIBC_TYPES_HAS_FLOAT128) &&
       // defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)

template <typename InputType, typename OutputType>
void explain_ternary_operation_one_output_error(
    Operation op, const TernaryInput<InputType> &input, OutputType libc_result,
    double ulp_tolerance, RoundingMode rounding) {
  unsigned int precision = get_precision<InputType>(ulp_tolerance);
  MPFRNumber mpfrX(input.x, precision);
  MPFRNumber mpfrY(input.y, precision);
  MPFRNumber mpfrZ(input.z, precision);
  FPBits<InputType> xbits(input.x);
  FPBits<InputType> ybits(input.y);
  FPBits<InputType> zbits(input.z);
  MPFRNumber mpfr_result = ternary_operation_one_output(
      op, input.x, input.y, input.z, precision, rounding);
  MPFRNumber mpfrMatchValue(libc_result);

  tlog << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str()
       << " z: " << mpfrZ.str() << '\n';
  tlog << " First input bits: " << str(FPBits<InputType>(input.x)) << '\n';
  tlog << "Second input bits: " << str(FPBits<InputType>(input.y)) << '\n';
  tlog << " Third input bits: " << str(FPBits<InputType>(input.z)) << '\n';

  tlog << "Libc result: " << mpfrMatchValue.str() << '\n'
       << "MPFR result: " << mpfr_result.str() << '\n';
  tlog << "Libc floating point result bits: "
       << str(FPBits<OutputType>(libc_result)) << '\n';
  tlog << "              MPFR rounded bits: "
       << str(FPBits<OutputType>(mpfr_result.as<OutputType>())) << '\n';
  tlog << "ULP error: " << mpfr_result.ulp_as_mpfr_number(libc_result).str()
       << '\n';
}

template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<float> &, float, double, RoundingMode);
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<double> &, float, double, RoundingMode);
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<double> &, double, double, RoundingMode);
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<long double> &, float, double, RoundingMode);
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<long double> &, double, double, RoundingMode);
template void
explain_ternary_operation_one_output_error(Operation,
                                           const TernaryInput<long double> &,
                                           long double, double, RoundingMode);

#ifdef LIBC_TYPES_HAS_FLOAT16
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<float16> &, float16, double, RoundingMode);
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<float> &, float16, double, RoundingMode);
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<double> &, float16, double, RoundingMode);
template void
explain_ternary_operation_one_output_error(Operation,
                                           const TernaryInput<long double> &,
                                           float16, double, RoundingMode);
#endif

template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<float> &, bfloat16, double, RoundingMode);
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<double> &, bfloat16, double, RoundingMode);
template void
explain_ternary_operation_one_output_error(Operation,
                                           const TernaryInput<long double> &,
                                           bfloat16, double, RoundingMode);
#if defined(LIBC_TYPES_HAS_FLOAT128) &&                                        \
    defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)
template void explain_ternary_operation_one_output_error(
    Operation, const TernaryInput<float128> &, bfloat16, double, RoundingMode);
#endif // defined(LIBC_TYPES_HAS_FLOAT128) &&
       // defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)

template <typename InputType, typename OutputType>
bool compare_unary_operation_single_output(Operation op, InputType input,
                                           OutputType libc_result,
                                           double ulp_tolerance,
                                           RoundingMode rounding) {
  unsigned int precision = get_precision<InputType>(ulp_tolerance);
  MPFRNumber mpfr_result;
  mpfr_result = unary_operation(op, input, precision, rounding);
  double ulp = mpfr_result.ulp(libc_result);
  return (ulp <= ulp_tolerance);
}

template bool compare_unary_operation_single_output(Operation, float, float,
                                                    double, RoundingMode);
template bool compare_unary_operation_single_output(Operation, double, double,
                                                    double, RoundingMode);
template bool compare_unary_operation_single_output(Operation, long double,
                                                    long double, double,
                                                    RoundingMode);
template bool compare_unary_operation_single_output(Operation, double, float,
                                                    double, RoundingMode);
template bool compare_unary_operation_single_output(Operation, long double,
                                                    float, double,
                                                    RoundingMode);
template bool compare_unary_operation_single_output(Operation, long double,
                                                    double, double,
                                                    RoundingMode);
#ifdef LIBC_TYPES_HAS_FLOAT16
template bool compare_unary_operation_single_output(Operation, float16, float16,
                                                    double, RoundingMode);
template bool compare_unary_operation_single_output(Operation, float, float16,
                                                    double, RoundingMode);
template bool compare_unary_operation_single_output(Operation, double, float16,
                                                    double, RoundingMode);
template bool compare_unary_operation_single_output(Operation, long double,
                                                    float16, double,
                                                    RoundingMode);
#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
template bool compare_unary_operation_single_output(Operation, float128,
                                                    float16, double,
                                                    RoundingMode);
#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
template bool compare_unary_operation_single_output(Operation, float128,
                                                    float128, double,
                                                    RoundingMode);
template bool compare_unary_operation_single_output(Operation, float128, float,
                                                    double, RoundingMode);
template bool compare_unary_operation_single_output(Operation, float128, double,
                                                    double, RoundingMode);
template bool compare_unary_operation_single_output(Operation, float128,
                                                    long double, double,
                                                    RoundingMode);
#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE

template <typename T>
bool compare_unary_operation_two_outputs(Operation op, T input,
                                         const BinaryOutput<T> &libc_result,
                                         double ulp_tolerance,
                                         RoundingMode rounding) {
  int mpfrIntResult;
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfr_result = unary_operation_two_outputs(op, input, mpfrIntResult,
                                                       precision, rounding);
  double ulp = mpfr_result.ulp(libc_result.f);

  if (mpfrIntResult != libc_result.i)
    return false;

  return (ulp <= ulp_tolerance);
}

template bool compare_unary_operation_two_outputs<float>(
    Operation, float, const BinaryOutput<float> &, double, RoundingMode);
template bool compare_unary_operation_two_outputs<double>(
    Operation, double, const BinaryOutput<double> &, double, RoundingMode);
template bool compare_unary_operation_two_outputs<long double>(
    Operation, long double, const BinaryOutput<long double> &, double,
    RoundingMode);

template <typename T>
bool compare_binary_operation_two_outputs(Operation op,
                                          const BinaryInput<T> &input,
                                          const BinaryOutput<T> &libc_result,
                                          double ulp_tolerance,
                                          RoundingMode rounding) {
  int mpfrIntResult;
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfr_result = binary_operation_two_outputs(
      op, input.x, input.y, mpfrIntResult, precision, rounding);
  double ulp = mpfr_result.ulp(libc_result.f);

  if (mpfrIntResult != libc_result.i) {
    if (op == Operation::RemQuo) {
      if ((0x7 & mpfrIntResult) != (0x7 & libc_result.i))
        return false;
    } else {
      return false;
    }
  }

  return (ulp <= ulp_tolerance);
}

template bool compare_binary_operation_two_outputs<float>(
    Operation, const BinaryInput<float> &, const BinaryOutput<float> &, double,
    RoundingMode);
template bool compare_binary_operation_two_outputs<double>(
    Operation, const BinaryInput<double> &, const BinaryOutput<double> &,
    double, RoundingMode);
template bool compare_binary_operation_two_outputs<long double>(
    Operation, const BinaryInput<long double> &,
    const BinaryOutput<long double> &, double, RoundingMode);

template <typename InputType, typename OutputType>
bool compare_binary_operation_one_output(Operation op,
                                         const BinaryInput<InputType> &input,
                                         OutputType libc_result,
                                         double ulp_tolerance,
                                         RoundingMode rounding) {
  unsigned int precision = get_precision<InputType>(ulp_tolerance);
  MPFRNumber mpfr_result =
      binary_operation_one_output(op, input.x, input.y, precision, rounding);
  double ulp = mpfr_result.ulp(libc_result);

  return (ulp <= ulp_tolerance);
}

template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<float> &,
                                                  float, double, RoundingMode);
template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<double> &,
                                                  double, double, RoundingMode);
template bool
compare_binary_operation_one_output(Operation, const BinaryInput<long double> &,
                                    float, double, RoundingMode);
template bool
compare_binary_operation_one_output(Operation, const BinaryInput<long double> &,
                                    double, double, RoundingMode);
template bool
compare_binary_operation_one_output(Operation, const BinaryInput<long double> &,
                                    long double, double, RoundingMode);

template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<double> &,
                                                  float, double, RoundingMode);
#ifdef LIBC_TYPES_HAS_FLOAT16
template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<float16> &,
                                                  float16, double,
                                                  RoundingMode);
template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<float> &,
                                                  float16, double,
                                                  RoundingMode);
template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<double> &,
                                                  float16, double,
                                                  RoundingMode);
template bool
compare_binary_operation_one_output(Operation, const BinaryInput<long double> &,
                                    float16, double, RoundingMode);
#endif
#if defined(LIBC_TYPES_HAS_FLOAT128) &&                                        \
    defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)
template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<float128> &,
                                                  float128, double,
                                                  RoundingMode);
#endif
template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<bfloat16> &,
                                                  bfloat16, double,
                                                  RoundingMode);

template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<float> &,
                                                  bfloat16, double,
                                                  RoundingMode);
template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<double> &,
                                                  bfloat16, double,
                                                  RoundingMode);
template bool
compare_binary_operation_one_output(Operation, const BinaryInput<long double> &,
                                    bfloat16, double, RoundingMode);
#if defined(LIBC_TYPES_HAS_FLOAT128) &&                                        \
    defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)
template bool compare_binary_operation_one_output(Operation,
                                                  const BinaryInput<float128> &,
                                                  bfloat16, double,
                                                  RoundingMode);
#endif // defined(LIBC_TYPES_HAS_FLOAT128) &&
       // defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)
template <typename InputType, typename OutputType>
bool compare_ternary_operation_one_output(Operation op,
                                          const TernaryInput<InputType> &input,
                                          OutputType libc_result,
                                          double ulp_tolerance,
                                          RoundingMode rounding) {
  unsigned int precision = get_precision<InputType>(ulp_tolerance);
  MPFRNumber mpfr_result = ternary_operation_one_output(
      op, input.x, input.y, input.z, precision, rounding);
  double ulp = mpfr_result.ulp(libc_result);

  return (ulp <= ulp_tolerance);
}

template bool compare_ternary_operation_one_output(Operation,
                                                   const TernaryInput<float> &,
                                                   float, double, RoundingMode);
template bool compare_ternary_operation_one_output(Operation,
                                                   const TernaryInput<double> &,
                                                   float, double, RoundingMode);
template bool compare_ternary_operation_one_output(Operation,
                                                   const TernaryInput<double> &,
                                                   double, double,
                                                   RoundingMode);
template bool compare_ternary_operation_one_output(
    Operation, const TernaryInput<long double> &, float, double, RoundingMode);
template bool compare_ternary_operation_one_output(
    Operation, const TernaryInput<long double> &, double, double, RoundingMode);
template bool
compare_ternary_operation_one_output(Operation,
                                     const TernaryInput<long double> &,
                                     long double, double, RoundingMode);

#ifdef LIBC_TYPES_HAS_FLOAT16
template bool
compare_ternary_operation_one_output(Operation, const TernaryInput<float16> &,
                                     float16, double, RoundingMode);
template bool compare_ternary_operation_one_output(Operation,
                                                   const TernaryInput<float> &,
                                                   float16, double,
                                                   RoundingMode);
template bool compare_ternary_operation_one_output(Operation,
                                                   const TernaryInput<double> &,
                                                   float16, double,
                                                   RoundingMode);
template bool
compare_ternary_operation_one_output(Operation,
                                     const TernaryInput<long double> &, float16,
                                     double, RoundingMode);
#endif

template bool compare_ternary_operation_one_output(Operation,
                                                   const TernaryInput<float> &,
                                                   bfloat16, double,
                                                   RoundingMode);
template bool compare_ternary_operation_one_output(Operation,
                                                   const TernaryInput<double> &,
                                                   bfloat16, double,
                                                   RoundingMode);
template bool
compare_ternary_operation_one_output(Operation,
                                     const TernaryInput<long double> &,
                                     bfloat16, double, RoundingMode);

#if defined(LIBC_TYPES_HAS_FLOAT128) &&                                        \
    defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)
template bool
compare_ternary_operation_one_output(Operation, const TernaryInput<float128> &,
                                     bfloat16, double, RoundingMode);
#endif // defined(LIBC_TYPES_HAS_FLOAT128) &&
       // defined(LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE)

} // namespace internal

template <typename T> bool round_to_long(T x, long &result) {
  MPFRNumber mpfr(x);
  return mpfr.round_to_long(result);
}

template bool round_to_long<float>(float, long &);
template bool round_to_long<double>(double, long &);
template bool round_to_long<long double>(long double, long &);

#ifdef LIBC_TYPES_HAS_FLOAT16
template bool round_to_long<float16>(float16, long &);
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
template bool round_to_long<float128>(float128, long &);
#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE

template bool round_to_long<bfloat16>(bfloat16, long &);

template <typename T> bool round_to_long(T x, RoundingMode mode, long &result) {
  MPFRNumber mpfr(x);
  return mpfr.round_to_long(get_mpfr_rounding_mode(mode), result);
}

template bool round_to_long<float>(float, RoundingMode, long &);
template bool round_to_long<double>(double, RoundingMode, long &);
template bool round_to_long<long double>(long double, RoundingMode, long &);

#ifdef LIBC_TYPES_HAS_FLOAT16
template bool round_to_long<float16>(float16, RoundingMode, long &);
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
template bool round_to_long<float128>(float128, RoundingMode, long &);
#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE

template bool round_to_long<bfloat16>(bfloat16, RoundingMode, long &);

template <typename T> T round(T x, RoundingMode mode) {
  MPFRNumber mpfr(x);
  MPFRNumber result = mpfr.rint(get_mpfr_rounding_mode(mode));
  return result.as<T>();
}

template float round<float>(float, RoundingMode);
template double round<double>(double, RoundingMode);
template long double round<long double>(long double, RoundingMode);

#ifdef LIBC_TYPES_HAS_FLOAT16
template float16 round<float16>(float16, RoundingMode);
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
template float128 round<float128>(float128, RoundingMode);
#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE

template bfloat16 round<bfloat16>(bfloat16, RoundingMode);

} // namespace mpfr
} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
