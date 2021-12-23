//===-- Utils which wrap MPFR ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPFRUtils.h"

#include "src/__support/CPP/StringView.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/architectures.h"
#include "utils/UnitTest/FPMatcher.h"

#include <cmath>
#include <memory>
#include <stdint.h>
#include <string>

#ifdef CUSTOM_MPFR_INCLUDER
// Some downstream repos are monoliths carrying MPFR sources in their third
// party directory. In such repos, including the MPFR header as
// `#include <mpfr.h>` is either disallowed or not possible. If that is the
// case, a file named `CustomMPFRIncluder.h` should be added through which the
// MPFR header can be included in manner allowed in that repo.
#include "CustomMPFRIncluder.h"
#else
#include <mpfr.h>
#endif

template <typename T> using FPBits = __llvm_libc::fputil::FPBits<T>;

namespace __llvm_libc {
namespace testing {
namespace mpfr {

template <typename T> struct Precision;

template <> struct Precision<float> {
  static constexpr unsigned int VALUE = 24;
};

template <> struct Precision<double> {
  static constexpr unsigned int VALUE = 53;
};

#if !(defined(LLVM_LIBC_ARCH_X86))
template <> struct Precision<long double> {
  static constexpr unsigned int VALUE = 64;
};
#else
template <> struct Precision<long double> {
  static constexpr unsigned int VALUE = 113;
};
#endif

class MPFRNumber {
  // A precision value which allows sufficiently large additional
  // precision even compared to quad-precision floating point values.
  unsigned int mpfr_precision;

  mpfr_t value;

public:
  MPFRNumber() : mpfr_precision(256) { mpfr_init2(value, mpfr_precision); }

  // We use explicit EnableIf specializations to disallow implicit
  // conversions. Implicit conversions can potentially lead to loss of
  // precision.
  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<float, XType>::Value, int> = 0>
  explicit MPFRNumber(XType x, int precision = 128)
      : mpfr_precision(precision) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set_flt(value, x, MPFR_RNDN);
  }

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<double, XType>::Value, int> = 0>
  explicit MPFRNumber(XType x, int precision = 128)
      : mpfr_precision(precision) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set_d(value, x, MPFR_RNDN);
  }

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<long double, XType>::Value, int> = 0>
  explicit MPFRNumber(XType x, int precision = 128)
      : mpfr_precision(precision) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set_ld(value, x, MPFR_RNDN);
  }

  template <typename XType,
            cpp::EnableIfType<cpp::IsIntegral<XType>::Value, int> = 0>
  explicit MPFRNumber(XType x, int precision = 128)
      : mpfr_precision(precision) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set_sj(value, x, MPFR_RNDN);
  }

  MPFRNumber(const MPFRNumber &other) : mpfr_precision(other.mpfr_precision) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set(value, other.value, MPFR_RNDN);
  }

  ~MPFRNumber() { mpfr_clear(value); }

  MPFRNumber &operator=(const MPFRNumber &rhs) {
    mpfr_precision = rhs.mpfr_precision;
    mpfr_set(value, rhs.value, MPFR_RNDN);
    return *this;
  }

  MPFRNumber abs() const {
    MPFRNumber result;
    mpfr_abs(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber ceil() const {
    MPFRNumber result;
    mpfr_ceil(result.value, value);
    return result;
  }

  MPFRNumber cos() const {
    MPFRNumber result;
    mpfr_cos(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber exp() const {
    MPFRNumber result;
    mpfr_exp(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber exp2() const {
    MPFRNumber result;
    mpfr_exp2(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber expm1() const {
    MPFRNumber result;
    mpfr_expm1(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber floor() const {
    MPFRNumber result;
    mpfr_floor(result.value, value);
    return result;
  }

  MPFRNumber frexp(int &exp) {
    MPFRNumber result;
    mpfr_exp_t resultExp;
    mpfr_frexp(&resultExp, result.value, value, MPFR_RNDN);
    exp = resultExp;
    return result;
  }

  MPFRNumber hypot(const MPFRNumber &b) {
    MPFRNumber result;
    mpfr_hypot(result.value, value, b.value, MPFR_RNDN);
    return result;
  }

  MPFRNumber log() const {
    MPFRNumber result;
    mpfr_log(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber remquo(const MPFRNumber &divisor, int &quotient) {
    MPFRNumber remainder;
    long q;
    mpfr_remquo(remainder.value, &q, value, divisor.value, MPFR_RNDN);
    quotient = q;
    return remainder;
  }

  MPFRNumber round() const {
    MPFRNumber result;
    mpfr_round(result.value, value);
    return result;
  }

  bool roung_to_long(long &result) const {
    // We first calculate the rounded value. This way, when converting
    // to long using mpfr_get_si, the rounding direction of MPFR_RNDN
    // (or any other rounding mode), does not have an influence.
    MPFRNumber roundedValue = round();
    mpfr_clear_erangeflag();
    result = mpfr_get_si(roundedValue.value, MPFR_RNDN);
    return mpfr_erangeflag_p();
  }

  bool roung_to_long(mpfr_rnd_t rnd, long &result) const {
    MPFRNumber rint_result;
    mpfr_rint(rint_result.value, value, rnd);
    return rint_result.roung_to_long(result);
  }

  MPFRNumber rint(mpfr_rnd_t rnd) const {
    MPFRNumber result;
    mpfr_rint(result.value, value, rnd);
    return result;
  }

  MPFRNumber mod_2pi() const {
    MPFRNumber result(0.0, 1280);
    MPFRNumber _2pi(0.0, 1280);
    mpfr_const_pi(_2pi.value, MPFR_RNDN);
    mpfr_mul_si(_2pi.value, _2pi.value, 2, MPFR_RNDN);
    mpfr_fmod(result.value, value, _2pi.value, MPFR_RNDN);
    return result;
  }

  MPFRNumber mod_pi_over_2() const {
    MPFRNumber result(0.0, 1280);
    MPFRNumber pi_over_2(0.0, 1280);
    mpfr_const_pi(pi_over_2.value, MPFR_RNDN);
    mpfr_mul_d(pi_over_2.value, pi_over_2.value, 0.5, MPFR_RNDN);
    mpfr_fmod(result.value, value, pi_over_2.value, MPFR_RNDN);
    return result;
  }

  MPFRNumber mod_pi_over_4() const {
    MPFRNumber result(0.0, 1280);
    MPFRNumber pi_over_4(0.0, 1280);
    mpfr_const_pi(pi_over_4.value, MPFR_RNDN);
    mpfr_mul_d(pi_over_4.value, pi_over_4.value, 0.25, MPFR_RNDN);
    mpfr_fmod(result.value, value, pi_over_4.value, MPFR_RNDN);
    return result;
  }

  MPFRNumber sin() const {
    MPFRNumber result;
    mpfr_sin(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber sqrt() const {
    MPFRNumber result;
    mpfr_sqrt(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber tan() const {
    MPFRNumber result;
    mpfr_tan(result.value, value, MPFR_RNDN);
    return result;
  }

  MPFRNumber trunc() const {
    MPFRNumber result;
    mpfr_trunc(result.value, value);
    return result;
  }

  MPFRNumber fma(const MPFRNumber &b, const MPFRNumber &c) {
    MPFRNumber result(*this);
    mpfr_fma(result.value, value, b.value, c.value, MPFR_RNDN);
    return result;
  }

  std::string str() const {
    // 200 bytes should be more than sufficient to hold a 100-digit number
    // plus additional bytes for the decimal point, '-' sign etc.
    constexpr size_t printBufSize = 200;
    char buffer[printBufSize];
    mpfr_snprintf(buffer, printBufSize, "%100.50Rf", value);
    cpp::StringView view(buffer);
    view = view.trim(' ');
    return std::string(view.data());
  }

  // These functions are useful for debugging.
  template <typename T> T as() const;

  template <> float as<float>() const { return mpfr_get_flt(value, MPFR_RNDN); }
  template <> double as<double>() const { return mpfr_get_d(value, MPFR_RNDN); }
  template <> long double as<long double>() const {
    return mpfr_get_ld(value, MPFR_RNDN);
  }

  void dump(const char *msg) const { mpfr_printf("%s%.128Rf\n", msg, value); }

  // Return the ULP (units-in-the-last-place) difference between the
  // stored MPFR and a floating point number.
  //
  // We define ULP difference as follows:
  //   If exponents of this value and the |input| are same, then:
  //     ULP(this_value, input) = abs(this_value - input) / eps(input)
  //   else:
  //     max = max(abs(this_value), abs(input))
  //     min = min(abs(this_value), abs(input))
  //     maxExponent = exponent(max)
  //     ULP(this_value, input) = (max - 2^maxExponent) / eps(max) +
  //                              (2^maxExponent - min) / eps(min)
  //
  // Remarks:
  // 1. A ULP of 0.0 will imply that the value is correctly rounded.
  // 2. We expect that this value and the value to be compared (the [input]
  //    argument) are reasonable close, and we will provide an upper bound
  //    of ULP value for testing.  Morever, most of the fractional parts of
  //    ULP value do not matter much, so using double as the return type
  //    should be good enough.
  // 3. For close enough values (values which don't diff in their exponent by
  //    not more than 1), a ULP difference of N indicates a bit distance
  //    of N between this number and [input].
  // 4. A values of +0.0 and -0.0 are treated as equal.
  template <typename T>
  cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, double> ulp(T input) {
    T thisAsT = as<T>();
    if (thisAsT == input)
      return T(0.0);

    int thisExponent = fputil::FPBits<T>(thisAsT).get_exponent();
    int inputExponent = fputil::FPBits<T>(input).get_exponent();
    // Adjust the exponents for denormal numbers.
    if (fputil::FPBits<T>(thisAsT).get_unbiased_exponent() == 0)
      ++thisExponent;
    if (fputil::FPBits<T>(input).get_unbiased_exponent() == 0)
      ++inputExponent;

    if (thisAsT * input < 0 || thisExponent == inputExponent) {
      MPFRNumber inputMPFR(input);
      mpfr_sub(inputMPFR.value, value, inputMPFR.value, MPFR_RNDN);
      mpfr_abs(inputMPFR.value, inputMPFR.value, MPFR_RNDN);
      mpfr_mul_2si(inputMPFR.value, inputMPFR.value,
                   -thisExponent + int(fputil::MantissaWidth<T>::VALUE),
                   MPFR_RNDN);
      return inputMPFR.as<double>();
    }

    // If the control reaches here, it means that this number and input are
    // of the same sign but different exponent. In such a case, ULP error is
    // calculated as sum of two parts.
    thisAsT = std::abs(thisAsT);
    input = std::abs(input);
    T min = thisAsT > input ? input : thisAsT;
    T max = thisAsT > input ? thisAsT : input;
    int minExponent = fputil::FPBits<T>(min).get_exponent();
    int maxExponent = fputil::FPBits<T>(max).get_exponent();
    // Adjust the exponents for denormal numbers.
    if (fputil::FPBits<T>(min).get_unbiased_exponent() == 0)
      ++minExponent;
    if (fputil::FPBits<T>(max).get_unbiased_exponent() == 0)
      ++maxExponent;

    MPFRNumber minMPFR(min);
    MPFRNumber maxMPFR(max);

    MPFRNumber pivot(uint32_t(1));
    mpfr_mul_2si(pivot.value, pivot.value, maxExponent, MPFR_RNDN);

    mpfr_sub(minMPFR.value, pivot.value, minMPFR.value, MPFR_RNDN);
    mpfr_mul_2si(minMPFR.value, minMPFR.value,
                 -minExponent + int(fputil::MantissaWidth<T>::VALUE),
                 MPFR_RNDN);

    mpfr_sub(maxMPFR.value, maxMPFR.value, pivot.value, MPFR_RNDN);
    mpfr_mul_2si(maxMPFR.value, maxMPFR.value,
                 -maxExponent + int(fputil::MantissaWidth<T>::VALUE),
                 MPFR_RNDN);

    mpfr_add(minMPFR.value, minMPFR.value, maxMPFR.value, MPFR_RNDN);
    return minMPFR.as<double>();
  }
};

namespace internal {

template <typename InputType>
cpp::EnableIfType<cpp::IsFloatingPointType<InputType>::Value, MPFRNumber>
unary_operation(Operation op, InputType input) {
  MPFRNumber mpfrInput(input);
  switch (op) {
  case Operation::Abs:
    return mpfrInput.abs();
  case Operation::Ceil:
    return mpfrInput.ceil();
  case Operation::Cos:
    return mpfrInput.cos();
  case Operation::Exp:
    return mpfrInput.exp();
  case Operation::Exp2:
    return mpfrInput.exp2();
  case Operation::Expm1:
    return mpfrInput.expm1();
  case Operation::Floor:
    return mpfrInput.floor();
  case Operation::Log:
    return mpfrInput.log();
  case Operation::Mod2PI:
    return mpfrInput.mod_2pi();
  case Operation::ModPIOver2:
    return mpfrInput.mod_pi_over_2();
  case Operation::ModPIOver4:
    return mpfrInput.mod_pi_over_4();
  case Operation::Round:
    return mpfrInput.round();
  case Operation::Sin:
    return mpfrInput.sin();
  case Operation::Sqrt:
    return mpfrInput.sqrt();
  case Operation::Tan:
    return mpfrInput.tan();
  case Operation::Trunc:
    return mpfrInput.trunc();
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::EnableIfType<cpp::IsFloatingPointType<InputType>::Value, MPFRNumber>
unary_operation_two_outputs(Operation op, InputType input, int &output) {
  MPFRNumber mpfrInput(input);
  switch (op) {
  case Operation::Frexp:
    return mpfrInput.frexp(output);
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::EnableIfType<cpp::IsFloatingPointType<InputType>::Value, MPFRNumber>
binary_operation_one_output(Operation op, InputType x, InputType y) {
  MPFRNumber inputX(x), inputY(y);
  switch (op) {
  case Operation::Hypot:
    return inputX.hypot(inputY);
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::EnableIfType<cpp::IsFloatingPointType<InputType>::Value, MPFRNumber>
binary_operation_two_outputs(Operation op, InputType x, InputType y,
                             int &output) {
  MPFRNumber inputX(x), inputY(y);
  switch (op) {
  case Operation::RemQuo:
    return inputX.remquo(inputY, output);
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::EnableIfType<cpp::IsFloatingPointType<InputType>::Value, MPFRNumber>
ternary_operation_one_output(Operation op, InputType x, InputType y,
                             InputType z) {
  // For FMA function, we just need to compare with the mpfr_fma with the same
  // precision as InputType.  Using higher precision as the intermediate results
  // to compare might incorrectly fail due to double-rounding errors.
  constexpr unsigned int prec = Precision<InputType>::VALUE;
  MPFRNumber inputX(x, prec), inputY(y, prec), inputZ(z, prec);
  switch (op) {
  case Operation::Fma:
    return inputX.fma(inputY, inputZ);
  default:
    __builtin_unreachable();
  }
}

template <typename T>
void explain_unary_operation_single_output_error(Operation op, T input,
                                                 T matchValue,
                                                 testutils::StreamWrapper &OS) {
  MPFRNumber mpfrInput(input);
  MPFRNumber mpfr_result = unary_operation(op, input);
  MPFRNumber mpfrMatchValue(matchValue);
  FPBits<T> inputBits(input);
  FPBits<T> matchBits(matchValue);
  FPBits<T> mpfr_resultBits(mpfr_result.as<T>());
  OS << "Match value not within tolerance value of MPFR result:\n"
     << "  Input decimal: " << mpfrInput.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("     Input bits: ", input, OS);
  OS << '\n' << "  Match decimal: " << mpfrMatchValue.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("     Match bits: ", matchValue,
                                              OS);
  OS << '\n' << "    MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "   MPFR rounded: ", mpfr_result.as<T>(), OS);
  OS << '\n';
  OS << "      ULP error: " << std::to_string(mpfr_result.ulp(matchValue))
     << '\n';
}

template void
explain_unary_operation_single_output_error<float>(Operation op, float, float,
                                                   testutils::StreamWrapper &);
template void explain_unary_operation_single_output_error<double>(
    Operation op, double, double, testutils::StreamWrapper &);
template void explain_unary_operation_single_output_error<long double>(
    Operation op, long double, long double, testutils::StreamWrapper &);

template <typename T>
void explain_unary_operation_two_outputs_error(
    Operation op, T input, const BinaryOutput<T> &libc_result,
    testutils::StreamWrapper &OS) {
  MPFRNumber mpfrInput(input);
  FPBits<T> inputBits(input);
  int mpfrIntResult;
  MPFRNumber mpfr_result =
      unary_operation_two_outputs(op, input, mpfrIntResult);

  if (mpfrIntResult != libc_result.i) {
    OS << "MPFR integral result: " << mpfrIntResult << '\n'
       << "Libc integral result: " << libc_result.i << '\n';
  } else {
    OS << "Integral result from libc matches integral result from MPFR.\n";
  }

  MPFRNumber mpfrMatchValue(libc_result.f);
  OS << "Libc floating point result is not within tolerance value of the MPFR "
     << "result.\n\n";

  OS << "            Input decimal: " << mpfrInput.str() << "\n\n";

  OS << "Libc floating point value: " << mpfrMatchValue.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      " Libc floating point bits: ", libc_result.f, OS);
  OS << "\n\n";

  OS << "              MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "             MPFR rounded: ", mpfr_result.as<T>(), OS);
  OS << '\n'
     << "                ULP error: "
     << std::to_string(mpfr_result.ulp(libc_result.f)) << '\n';
}

template void explain_unary_operation_two_outputs_error<float>(
    Operation, float, const BinaryOutput<float> &, testutils::StreamWrapper &);
template void
explain_unary_operation_two_outputs_error<double>(Operation, double,
                                                  const BinaryOutput<double> &,
                                                  testutils::StreamWrapper &);
template void explain_unary_operation_two_outputs_error<long double>(
    Operation, long double, const BinaryOutput<long double> &,
    testutils::StreamWrapper &);

template <typename T>
void explain_binary_operation_two_outputs_error(
    Operation op, const BinaryInput<T> &input,
    const BinaryOutput<T> &libc_result, testutils::StreamWrapper &OS) {
  MPFRNumber mpfrX(input.x);
  MPFRNumber mpfrY(input.y);
  FPBits<T> xbits(input.x);
  FPBits<T> ybits(input.y);
  int mpfrIntResult;
  MPFRNumber mpfr_result =
      binary_operation_two_outputs(op, input.x, input.y, mpfrIntResult);
  MPFRNumber mpfrMatchValue(libc_result.f);

  OS << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str() << '\n'
     << "MPFR integral result: " << mpfrIntResult << '\n'
     << "Libc integral result: " << libc_result.i << '\n'
     << "Libc floating point result: " << mpfrMatchValue.str() << '\n'
     << "               MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "Libc floating point result bits: ", libc_result.f, OS);
  __llvm_libc::fputil::testing::describeValue(
      "              MPFR rounded bits: ", mpfr_result.as<T>(), OS);
  OS << "ULP error: " << std::to_string(mpfr_result.ulp(libc_result.f)) << '\n';
}

template void explain_binary_operation_two_outputs_error<float>(
    Operation, const BinaryInput<float> &, const BinaryOutput<float> &,
    testutils::StreamWrapper &);
template void explain_binary_operation_two_outputs_error<double>(
    Operation, const BinaryInput<double> &, const BinaryOutput<double> &,
    testutils::StreamWrapper &);
template void explain_binary_operation_two_outputs_error<long double>(
    Operation, const BinaryInput<long double> &,
    const BinaryOutput<long double> &, testutils::StreamWrapper &);

template <typename T>
void explain_binary_operation_one_output_error(Operation op,
                                               const BinaryInput<T> &input,
                                               T libc_result,
                                               testutils::StreamWrapper &OS) {
  MPFRNumber mpfrX(input.x);
  MPFRNumber mpfrY(input.y);
  FPBits<T> xbits(input.x);
  FPBits<T> ybits(input.y);
  MPFRNumber mpfr_result = binary_operation_one_output(op, input.x, input.y);
  MPFRNumber mpfrMatchValue(libc_result);

  OS << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("First input bits: ", input.x,
                                              OS);
  __llvm_libc::fputil::testing::describeValue("Second input bits: ", input.y,
                                              OS);

  OS << "Libc result: " << mpfrMatchValue.str() << '\n'
     << "MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "Libc floating point result bits: ", libc_result, OS);
  __llvm_libc::fputil::testing::describeValue(
      "              MPFR rounded bits: ", mpfr_result.as<T>(), OS);
  OS << "ULP error: " << std::to_string(mpfr_result.ulp(libc_result)) << '\n';
}

template void explain_binary_operation_one_output_error<float>(
    Operation, const BinaryInput<float> &, float, testutils::StreamWrapper &);
template void explain_binary_operation_one_output_error<double>(
    Operation, const BinaryInput<double> &, double, testutils::StreamWrapper &);
template void explain_binary_operation_one_output_error<long double>(
    Operation, const BinaryInput<long double> &, long double,
    testutils::StreamWrapper &);

template <typename T>
void explain_ternary_operation_one_output_error(Operation op,
                                                const TernaryInput<T> &input,
                                                T libc_result,
                                                testutils::StreamWrapper &OS) {
  MPFRNumber mpfrX(input.x, Precision<T>::VALUE);
  MPFRNumber mpfrY(input.y, Precision<T>::VALUE);
  MPFRNumber mpfrZ(input.z, Precision<T>::VALUE);
  FPBits<T> xbits(input.x);
  FPBits<T> ybits(input.y);
  FPBits<T> zbits(input.z);
  MPFRNumber mpfr_result =
      ternary_operation_one_output(op, input.x, input.y, input.z);
  MPFRNumber mpfrMatchValue(libc_result);

  OS << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str()
     << " z: " << mpfrZ.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("First input bits: ", input.x,
                                              OS);
  __llvm_libc::fputil::testing::describeValue("Second input bits: ", input.y,
                                              OS);
  __llvm_libc::fputil::testing::describeValue("Third input bits: ", input.z,
                                              OS);

  OS << "Libc result: " << mpfrMatchValue.str() << '\n'
     << "MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "Libc floating point result bits: ", libc_result, OS);
  __llvm_libc::fputil::testing::describeValue(
      "              MPFR rounded bits: ", mpfr_result.as<T>(), OS);
  OS << "ULP error: " << std::to_string(mpfr_result.ulp(libc_result)) << '\n';
}

template void explain_ternary_operation_one_output_error<float>(
    Operation, const TernaryInput<float> &, float, testutils::StreamWrapper &);
template void explain_ternary_operation_one_output_error<double>(
    Operation, const TernaryInput<double> &, double,
    testutils::StreamWrapper &);
template void explain_ternary_operation_one_output_error<long double>(
    Operation, const TernaryInput<long double> &, long double,
    testutils::StreamWrapper &);

template <typename T>
bool compare_unary_operation_single_output(Operation op, T input, T libc_result,
                                           double ulp_error) {
  // If the ulp error is exactly 0.5 (i.e a tie), we would check that the result
  // is rounded to the nearest even.
  MPFRNumber mpfr_result = unary_operation(op, input);
  double ulp = mpfr_result.ulp(libc_result);
  bool bits_are_even = ((FPBits<T>(libc_result).uintval() & 1) == 0);
  return (ulp < ulp_error) ||
         ((ulp == ulp_error) && ((ulp != 0.5) || bits_are_even));
}

template bool compare_unary_operation_single_output<float>(Operation, float,
                                                           float, double);
template bool compare_unary_operation_single_output<double>(Operation, double,
                                                            double, double);
template bool compare_unary_operation_single_output<long double>(Operation,
                                                                 long double,
                                                                 long double,
                                                                 double);

template <typename T>
bool compare_unary_operation_two_outputs(Operation op, T input,
                                         const BinaryOutput<T> &libc_result,
                                         double ulp_error) {
  int mpfrIntResult;
  MPFRNumber mpfr_result =
      unary_operation_two_outputs(op, input, mpfrIntResult);
  double ulp = mpfr_result.ulp(libc_result.f);

  if (mpfrIntResult != libc_result.i)
    return false;

  bool bits_are_even = ((FPBits<T>(libc_result.f).uintval() & 1) == 0);
  return (ulp < ulp_error) ||
         ((ulp == ulp_error) && ((ulp != 0.5) || bits_are_even));
}

template bool
compare_unary_operation_two_outputs<float>(Operation, float,
                                           const BinaryOutput<float> &, double);
template bool compare_unary_operation_two_outputs<double>(
    Operation, double, const BinaryOutput<double> &, double);
template bool compare_unary_operation_two_outputs<long double>(
    Operation, long double, const BinaryOutput<long double> &, double);

template <typename T>
bool compare_binary_operation_two_outputs(Operation op,
                                          const BinaryInput<T> &input,
                                          const BinaryOutput<T> &libc_result,
                                          double ulp_error) {
  int mpfrIntResult;
  MPFRNumber mpfr_result =
      binary_operation_two_outputs(op, input.x, input.y, mpfrIntResult);
  double ulp = mpfr_result.ulp(libc_result.f);

  if (mpfrIntResult != libc_result.i) {
    if (op == Operation::RemQuo) {
      if ((0x7 & mpfrIntResult) != (0x7 & libc_result.i))
        return false;
    } else {
      return false;
    }
  }

  bool bits_are_even = ((FPBits<T>(libc_result.f).uintval() & 1) == 0);
  return (ulp < ulp_error) ||
         ((ulp == ulp_error) && ((ulp != 0.5) || bits_are_even));
}

template bool compare_binary_operation_two_outputs<float>(
    Operation, const BinaryInput<float> &, const BinaryOutput<float> &, double);
template bool compare_binary_operation_two_outputs<double>(
    Operation, const BinaryInput<double> &, const BinaryOutput<double> &,
    double);
template bool compare_binary_operation_two_outputs<long double>(
    Operation, const BinaryInput<long double> &,
    const BinaryOutput<long double> &, double);

template <typename T>
bool compare_binary_operation_one_output(Operation op,
                                         const BinaryInput<T> &input,
                                         T libc_result, double ulp_error) {
  MPFRNumber mpfr_result = binary_operation_one_output(op, input.x, input.y);
  double ulp = mpfr_result.ulp(libc_result);

  bool bits_are_even = ((FPBits<T>(libc_result).uintval() & 1) == 0);
  return (ulp < ulp_error) ||
         ((ulp == ulp_error) && ((ulp != 0.5) || bits_are_even));
}

template bool compare_binary_operation_one_output<float>(
    Operation, const BinaryInput<float> &, float, double);
template bool compare_binary_operation_one_output<double>(
    Operation, const BinaryInput<double> &, double, double);
template bool compare_binary_operation_one_output<long double>(
    Operation, const BinaryInput<long double> &, long double, double);

template <typename T>
bool compare_ternary_operation_one_output(Operation op,
                                          const TernaryInput<T> &input,
                                          T libc_result, double ulp_error) {
  MPFRNumber mpfr_result =
      ternary_operation_one_output(op, input.x, input.y, input.z);
  double ulp = mpfr_result.ulp(libc_result);

  bool bits_are_even = ((FPBits<T>(libc_result).uintval() & 1) == 0);
  return (ulp < ulp_error) ||
         ((ulp == ulp_error) && ((ulp != 0.5) || bits_are_even));
}

template bool compare_ternary_operation_one_output<float>(
    Operation, const TernaryInput<float> &, float, double);
template bool compare_ternary_operation_one_output<double>(
    Operation, const TernaryInput<double> &, double, double);
template bool compare_ternary_operation_one_output<long double>(
    Operation, const TernaryInput<long double> &, long double, double);

static mpfr_rnd_t get_mpfr_rounding_mode(RoundingMode mode) {
  switch (mode) {
  case RoundingMode::Upward:
    return MPFR_RNDU;
    break;
  case RoundingMode::Downward:
    return MPFR_RNDD;
    break;
  case RoundingMode::TowardZero:
    return MPFR_RNDZ;
    break;
  case RoundingMode::Nearest:
    return MPFR_RNDN;
    break;
  }
}

} // namespace internal

template <typename T> bool round_to_long(T x, long &result) {
  MPFRNumber mpfr(x);
  return mpfr.roung_to_long(result);
}

template bool round_to_long<float>(float, long &);
template bool round_to_long<double>(double, long &);
template bool round_to_long<long double>(long double, long &);

template <typename T> bool round_to_long(T x, RoundingMode mode, long &result) {
  MPFRNumber mpfr(x);
  return mpfr.roung_to_long(internal::get_mpfr_rounding_mode(mode), result);
}

template bool round_to_long<float>(float, RoundingMode, long &);
template bool round_to_long<double>(double, RoundingMode, long &);
template bool round_to_long<long double>(long double, RoundingMode, long &);

template <typename T> T round(T x, RoundingMode mode) {
  MPFRNumber mpfr(x);
  MPFRNumber result = mpfr.rint(internal::get_mpfr_rounding_mode(mode));
  return result.as<T>();
}

template float round<float>(float, RoundingMode);
template double round<double>(double, RoundingMode);
template long double round<long double>(long double, RoundingMode);

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc
