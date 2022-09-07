//===-- Utils which wrap MPFR ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPFRUtils.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PlatformDefs.h"
#include "utils/UnitTest/FPMatcher.h"

#include <cmath>
#include <fenv.h>
#include <memory>
#include <sstream>
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

#if defined(LONG_DOUBLE_IS_DOUBLE)
template <> struct Precision<long double> {
  static constexpr unsigned int VALUE = 53;
};
#elif defined(SPECIAL_X86_LONG_DOUBLE)
template <> struct Precision<long double> {
  static constexpr unsigned int VALUE = 64;
};
#else
template <> struct Precision<long double> {
  static constexpr unsigned int VALUE = 113;
};
#endif

// A precision value which allows sufficiently large additional
// precision compared to the floating point precision.
template <typename T> struct ExtraPrecision;

template <> struct ExtraPrecision<float> {
  static constexpr unsigned int VALUE = 128;
};

template <> struct ExtraPrecision<double> {
  static constexpr unsigned int VALUE = 256;
};

template <> struct ExtraPrecision<long double> {
  static constexpr unsigned int VALUE = 256;
};

// If the ulp tolerance is less than or equal to 0.5, we would check that the
// result is rounded correctly with respect to the rounding mode by using the
// same precision as the inputs.
template <typename T>
static inline unsigned int get_precision(double ulp_tolerance) {
  if (ulp_tolerance <= 0.5) {
    return Precision<T>::VALUE;
  } else {
    return ExtraPrecision<T>::VALUE;
  }
}

static inline mpfr_rnd_t get_mpfr_rounding_mode(RoundingMode mode) {
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

class MPFRNumber {
  unsigned int mpfr_precision;
  mpfr_rnd_t mpfr_rounding;

  mpfr_t value;

public:
  MPFRNumber() : mpfr_precision(256), mpfr_rounding(MPFR_RNDN) {
    mpfr_init2(value, mpfr_precision);
  }

  // We use explicit EnableIf specializations to disallow implicit
  // conversions. Implicit conversions can potentially lead to loss of
  // precision.
  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<float, XType>, int> = 0>
  explicit MPFRNumber(XType x, int precision = ExtraPrecision<XType>::VALUE,
                      RoundingMode rounding = RoundingMode::Nearest)
      : mpfr_precision(precision),
        mpfr_rounding(get_mpfr_rounding_mode(rounding)) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set_flt(value, x, mpfr_rounding);
  }

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<double, XType>, int> = 0>
  explicit MPFRNumber(XType x, int precision = ExtraPrecision<XType>::VALUE,
                      RoundingMode rounding = RoundingMode::Nearest)
      : mpfr_precision(precision),
        mpfr_rounding(get_mpfr_rounding_mode(rounding)) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set_d(value, x, mpfr_rounding);
  }

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<long double, XType>, int> = 0>
  explicit MPFRNumber(XType x, int precision = ExtraPrecision<XType>::VALUE,
                      RoundingMode rounding = RoundingMode::Nearest)
      : mpfr_precision(precision),
        mpfr_rounding(get_mpfr_rounding_mode(rounding)) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set_ld(value, x, mpfr_rounding);
  }

  template <typename XType,
            cpp::enable_if_t<cpp::is_integral_v<XType>, int> = 0>
  explicit MPFRNumber(XType x, int precision = ExtraPrecision<float>::VALUE,
                      RoundingMode rounding = RoundingMode::Nearest)
      : mpfr_precision(precision),
        mpfr_rounding(get_mpfr_rounding_mode(rounding)) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set_sj(value, x, mpfr_rounding);
  }

  MPFRNumber(const MPFRNumber &other)
      : mpfr_precision(other.mpfr_precision),
        mpfr_rounding(other.mpfr_rounding) {
    mpfr_init2(value, mpfr_precision);
    mpfr_set(value, other.value, mpfr_rounding);
  }

  ~MPFRNumber() { mpfr_clear(value); }

  MPFRNumber &operator=(const MPFRNumber &rhs) {
    mpfr_precision = rhs.mpfr_precision;
    mpfr_rounding = rhs.mpfr_rounding;
    mpfr_set(value, rhs.value, mpfr_rounding);
    return *this;
  }

  bool is_nan() const { return mpfr_nan_p(value); }

  MPFRNumber abs() const {
    MPFRNumber result(*this);
    mpfr_abs(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber asin() const {
    MPFRNumber result(*this);
    mpfr_asin(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber atan() const {
    MPFRNumber result(*this);
    mpfr_atan(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber atanh() const {
    MPFRNumber result(*this);
    mpfr_atanh(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber ceil() const {
    MPFRNumber result(*this);
    mpfr_ceil(result.value, value);
    return result;
  }

  MPFRNumber cos() const {
    MPFRNumber result(*this);
    mpfr_cos(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber cosh() const {
    MPFRNumber result(*this);
    mpfr_cosh(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber exp() const {
    MPFRNumber result(*this);
    mpfr_exp(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber exp2() const {
    MPFRNumber result(*this);
    mpfr_exp2(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber expm1() const {
    MPFRNumber result(*this);
    mpfr_expm1(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber floor() const {
    MPFRNumber result(*this);
    mpfr_floor(result.value, value);
    return result;
  }

  MPFRNumber fmod(const MPFRNumber &b) {
    MPFRNumber result(*this);
    mpfr_fmod(result.value, value, b.value, mpfr_rounding);
    return result;
  }

  MPFRNumber frexp(int &exp) {
    MPFRNumber result(*this);
    mpfr_exp_t resultExp;
    mpfr_frexp(&resultExp, result.value, value, mpfr_rounding);
    exp = resultExp;
    return result;
  }

  MPFRNumber hypot(const MPFRNumber &b) {
    MPFRNumber result(*this);
    mpfr_hypot(result.value, value, b.value, mpfr_rounding);
    return result;
  }

  MPFRNumber log() const {
    MPFRNumber result(*this);
    mpfr_log(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber log2() const {
    MPFRNumber result(*this);
    mpfr_log2(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber log10() const {
    MPFRNumber result(*this);
    mpfr_log10(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber log1p() const {
    MPFRNumber result(*this);
    mpfr_log1p(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber remquo(const MPFRNumber &divisor, int &quotient) {
    MPFRNumber remainder(*this);
    long q;
    mpfr_remquo(remainder.value, &q, value, divisor.value, mpfr_rounding);
    quotient = q;
    return remainder;
  }

  MPFRNumber round() const {
    MPFRNumber result(*this);
    mpfr_round(result.value, value);
    return result;
  }

  bool round_to_long(long &result) const {
    // We first calculate the rounded value. This way, when converting
    // to long using mpfr_get_si, the rounding direction of MPFR_RNDN
    // (or any other rounding mode), does not have an influence.
    MPFRNumber roundedValue = round();
    mpfr_clear_erangeflag();
    result = mpfr_get_si(roundedValue.value, MPFR_RNDN);
    return mpfr_erangeflag_p();
  }

  bool round_to_long(mpfr_rnd_t rnd, long &result) const {
    MPFRNumber rint_result(*this);
    mpfr_rint(rint_result.value, value, rnd);
    return rint_result.round_to_long(result);
  }

  MPFRNumber rint(mpfr_rnd_t rnd) const {
    MPFRNumber result(*this);
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
    MPFRNumber result(*this);
    mpfr_sin(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber sinh() const {
    MPFRNumber result(*this);
    mpfr_sinh(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber sqrt() const {
    MPFRNumber result(*this);
    mpfr_sqrt(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber tan() const {
    MPFRNumber result(*this);
    mpfr_tan(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber tanh() const {
    MPFRNumber result(*this);
    mpfr_tanh(result.value, value, mpfr_rounding);
    return result;
  }

  MPFRNumber trunc() const {
    MPFRNumber result(*this);
    mpfr_trunc(result.value, value);
    return result;
  }

  MPFRNumber fma(const MPFRNumber &b, const MPFRNumber &c) {
    MPFRNumber result(*this);
    mpfr_fma(result.value, value, b.value, c.value, mpfr_rounding);
    return result;
  }

  std::string str() const {
    // 200 bytes should be more than sufficient to hold a 100-digit number
    // plus additional bytes for the decimal point, '-' sign etc.
    constexpr size_t printBufSize = 200;
    char buffer[printBufSize];
    mpfr_snprintf(buffer, printBufSize, "%100.50Rf", value);
    cpp::string_view view(buffer);
    // Trim whitespaces
    const char whitespace = ' ';
    while (!view.empty() && view.front() == whitespace)
      view.remove_prefix(1);
    while (!view.empty() && view.back() == whitespace)
      view.remove_suffix(1);
    return std::string(view.data());
  }

  // These functions are useful for debugging.
  template <typename T> T as() const;

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
  cpp::enable_if_t<cpp::is_floating_point_v<T>, double> ulp(T input) {
    T thisAsT = as<T>();
    if (thisAsT == input)
      return T(0.0);

    if (is_nan()) {
      if (fputil::FPBits<T>(input).is_nan())
        return T(0.0);
      return T(fputil::FPBits<T>::inf());
    }

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

template <> float MPFRNumber::as<float>() const {
  return mpfr_get_flt(value, mpfr_rounding);
}

template <> double MPFRNumber::as<double>() const {
  return mpfr_get_d(value, mpfr_rounding);
}

template <> long double MPFRNumber::as<long double>() const {
  return mpfr_get_ld(value, mpfr_rounding);
}

namespace internal {

template <typename InputType>
cpp::enable_if_t<cpp::is_floating_point_v<InputType>, MPFRNumber>
unary_operation(Operation op, InputType input, unsigned int precision,
                RoundingMode rounding) {
  MPFRNumber mpfrInput(input, precision, rounding);
  switch (op) {
  case Operation::Abs:
    return mpfrInput.abs();
  case Operation::Asin:
    return mpfrInput.asin();
  case Operation::Atan:
    return mpfrInput.atan();
  case Operation::Atanh:
    return mpfrInput.atanh();
  case Operation::Ceil:
    return mpfrInput.ceil();
  case Operation::Cos:
    return mpfrInput.cos();
  case Operation::Cosh:
    return mpfrInput.cosh();
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
  case Operation::Sin:
    return mpfrInput.sin();
  case Operation::Sinh:
    return mpfrInput.sinh();
  case Operation::Sqrt:
    return mpfrInput.sqrt();
  case Operation::Tan:
    return mpfrInput.tan();
  case Operation::Tanh:
    return mpfrInput.tanh();
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
  case Operation::Fmod:
    return inputX.fmod(inputY);
  case Operation::Hypot:
    return inputX.hypot(inputY);
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
template <typename T>
void explain_unary_operation_single_output_error(Operation op, T input,
                                                 T matchValue,
                                                 double ulp_tolerance,
                                                 RoundingMode rounding,
                                                 testutils::StreamWrapper &OS) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfrInput(input, precision);
  MPFRNumber mpfr_result;
  mpfr_result = unary_operation(op, input, precision, rounding);
  MPFRNumber mpfrMatchValue(matchValue);
  std::stringstream ss;
  ss << "Match value not within tolerance value of MPFR result:\n"
     << "  Input decimal: " << mpfrInput.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("     Input bits: ", input, ss);
  ss << '\n' << "  Match decimal: " << mpfrMatchValue.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("     Match bits: ", matchValue,
                                              ss);
  ss << '\n' << "    MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "   MPFR rounded: ", mpfr_result.as<T>(), ss);
  ss << '\n';
  ss << "      ULP error: " << std::to_string(mpfr_result.ulp(matchValue))
     << '\n';
  OS << ss.str();
}

template void
explain_unary_operation_single_output_error<float>(Operation op, float, float,
                                                   double, RoundingMode,
                                                   testutils::StreamWrapper &);
template void explain_unary_operation_single_output_error<double>(
    Operation op, double, double, double, RoundingMode,
    testutils::StreamWrapper &);
template void explain_unary_operation_single_output_error<long double>(
    Operation op, long double, long double, double, RoundingMode,
    testutils::StreamWrapper &);

template <typename T>
void explain_unary_operation_two_outputs_error(
    Operation op, T input, const BinaryOutput<T> &libc_result,
    double ulp_tolerance, RoundingMode rounding, testutils::StreamWrapper &OS) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfrInput(input, precision);
  int mpfrIntResult;
  MPFRNumber mpfr_result = unary_operation_two_outputs(op, input, mpfrIntResult,
                                                       precision, rounding);
  std::stringstream ss;

  if (mpfrIntResult != libc_result.i) {
    ss << "MPFR integral result: " << mpfrIntResult << '\n'
       << "Libc integral result: " << libc_result.i << '\n';
  } else {
    ss << "Integral result from libc matches integral result from MPFR.\n";
  }

  MPFRNumber mpfrMatchValue(libc_result.f);
  ss << "Libc floating point result is not within tolerance value of the MPFR "
     << "result.\n\n";

  ss << "            Input decimal: " << mpfrInput.str() << "\n\n";

  ss << "Libc floating point value: " << mpfrMatchValue.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      " Libc floating point bits: ", libc_result.f, ss);
  ss << "\n\n";

  ss << "              MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "             MPFR rounded: ", mpfr_result.as<T>(), ss);
  ss << '\n'
     << "                ULP error: "
     << std::to_string(mpfr_result.ulp(libc_result.f)) << '\n';
  OS << ss.str();
}

template void explain_unary_operation_two_outputs_error<float>(
    Operation, float, const BinaryOutput<float> &, double, RoundingMode,
    testutils::StreamWrapper &);
template void explain_unary_operation_two_outputs_error<double>(
    Operation, double, const BinaryOutput<double> &, double, RoundingMode,
    testutils::StreamWrapper &);
template void explain_unary_operation_two_outputs_error<long double>(
    Operation, long double, const BinaryOutput<long double> &, double,
    RoundingMode, testutils::StreamWrapper &);

template <typename T>
void explain_binary_operation_two_outputs_error(
    Operation op, const BinaryInput<T> &input,
    const BinaryOutput<T> &libc_result, double ulp_tolerance,
    RoundingMode rounding, testutils::StreamWrapper &OS) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfrX(input.x, precision);
  MPFRNumber mpfrY(input.y, precision);
  int mpfrIntResult;
  MPFRNumber mpfr_result = binary_operation_two_outputs(
      op, input.x, input.y, mpfrIntResult, precision, rounding);
  MPFRNumber mpfrMatchValue(libc_result.f);
  std::stringstream ss;

  ss << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str() << '\n'
     << "MPFR integral result: " << mpfrIntResult << '\n'
     << "Libc integral result: " << libc_result.i << '\n'
     << "Libc floating point result: " << mpfrMatchValue.str() << '\n'
     << "               MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "Libc floating point result bits: ", libc_result.f, ss);
  __llvm_libc::fputil::testing::describeValue(
      "              MPFR rounded bits: ", mpfr_result.as<T>(), ss);
  ss << "ULP error: " << std::to_string(mpfr_result.ulp(libc_result.f)) << '\n';
  OS << ss.str();
}

template void explain_binary_operation_two_outputs_error<float>(
    Operation, const BinaryInput<float> &, const BinaryOutput<float> &, double,
    RoundingMode, testutils::StreamWrapper &);
template void explain_binary_operation_two_outputs_error<double>(
    Operation, const BinaryInput<double> &, const BinaryOutput<double> &,
    double, RoundingMode, testutils::StreamWrapper &);
template void explain_binary_operation_two_outputs_error<long double>(
    Operation, const BinaryInput<long double> &,
    const BinaryOutput<long double> &, double, RoundingMode,
    testutils::StreamWrapper &);

template <typename T>
void explain_binary_operation_one_output_error(
    Operation op, const BinaryInput<T> &input, T libc_result,
    double ulp_tolerance, RoundingMode rounding, testutils::StreamWrapper &OS) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfrX(input.x, precision);
  MPFRNumber mpfrY(input.y, precision);
  FPBits<T> xbits(input.x);
  FPBits<T> ybits(input.y);
  MPFRNumber mpfr_result =
      binary_operation_one_output(op, input.x, input.y, precision, rounding);
  MPFRNumber mpfrMatchValue(libc_result);
  std::stringstream ss;

  ss << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("First input bits: ", input.x,
                                              ss);
  __llvm_libc::fputil::testing::describeValue("Second input bits: ", input.y,
                                              ss);

  ss << "Libc result: " << mpfrMatchValue.str() << '\n'
     << "MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "Libc floating point result bits: ", libc_result, ss);
  __llvm_libc::fputil::testing::describeValue(
      "              MPFR rounded bits: ", mpfr_result.as<T>(), ss);
  ss << "ULP error: " << std::to_string(mpfr_result.ulp(libc_result)) << '\n';
  OS << ss.str();
}

template void explain_binary_operation_one_output_error<float>(
    Operation, const BinaryInput<float> &, float, double, RoundingMode,
    testutils::StreamWrapper &);
template void explain_binary_operation_one_output_error<double>(
    Operation, const BinaryInput<double> &, double, double, RoundingMode,
    testutils::StreamWrapper &);
template void explain_binary_operation_one_output_error<long double>(
    Operation, const BinaryInput<long double> &, long double, double,
    RoundingMode, testutils::StreamWrapper &);

template <typename T>
void explain_ternary_operation_one_output_error(
    Operation op, const TernaryInput<T> &input, T libc_result,
    double ulp_tolerance, RoundingMode rounding, testutils::StreamWrapper &OS) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfrX(input.x, precision);
  MPFRNumber mpfrY(input.y, precision);
  MPFRNumber mpfrZ(input.z, precision);
  FPBits<T> xbits(input.x);
  FPBits<T> ybits(input.y);
  FPBits<T> zbits(input.z);
  MPFRNumber mpfr_result = ternary_operation_one_output(
      op, input.x, input.y, input.z, precision, rounding);
  MPFRNumber mpfrMatchValue(libc_result);
  std::stringstream ss;

  ss << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str()
     << " z: " << mpfrZ.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("First input bits: ", input.x,
                                              ss);
  __llvm_libc::fputil::testing::describeValue("Second input bits: ", input.y,
                                              ss);
  __llvm_libc::fputil::testing::describeValue("Third input bits: ", input.z,
                                              ss);

  ss << "Libc result: " << mpfrMatchValue.str() << '\n'
     << "MPFR result: " << mpfr_result.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "Libc floating point result bits: ", libc_result, ss);
  __llvm_libc::fputil::testing::describeValue(
      "              MPFR rounded bits: ", mpfr_result.as<T>(), ss);
  ss << "ULP error: " << std::to_string(mpfr_result.ulp(libc_result)) << '\n';
  OS << ss.str();
}

template void explain_ternary_operation_one_output_error<float>(
    Operation, const TernaryInput<float> &, float, double, RoundingMode,
    testutils::StreamWrapper &);
template void explain_ternary_operation_one_output_error<double>(
    Operation, const TernaryInput<double> &, double, double, RoundingMode,
    testutils::StreamWrapper &);
template void explain_ternary_operation_one_output_error<long double>(
    Operation, const TernaryInput<long double> &, long double, double,
    RoundingMode, testutils::StreamWrapper &);

template <typename T>
bool compare_unary_operation_single_output(Operation op, T input, T libc_result,
                                           double ulp_tolerance,
                                           RoundingMode rounding) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfr_result;
  mpfr_result = unary_operation(op, input, precision, rounding);
  double ulp = mpfr_result.ulp(libc_result);
  return (ulp <= ulp_tolerance);
}

template bool compare_unary_operation_single_output<float>(Operation, float,
                                                           float, double,
                                                           RoundingMode);
template bool compare_unary_operation_single_output<double>(Operation, double,
                                                            double, double,
                                                            RoundingMode);
template bool compare_unary_operation_single_output<long double>(
    Operation, long double, long double, double, RoundingMode);

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

template <typename T>
bool compare_binary_operation_one_output(Operation op,
                                         const BinaryInput<T> &input,
                                         T libc_result, double ulp_tolerance,
                                         RoundingMode rounding) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfr_result =
      binary_operation_one_output(op, input.x, input.y, precision, rounding);
  double ulp = mpfr_result.ulp(libc_result);

  return (ulp <= ulp_tolerance);
}

template bool compare_binary_operation_one_output<float>(
    Operation, const BinaryInput<float> &, float, double, RoundingMode);
template bool compare_binary_operation_one_output<double>(
    Operation, const BinaryInput<double> &, double, double, RoundingMode);
template bool compare_binary_operation_one_output<long double>(
    Operation, const BinaryInput<long double> &, long double, double,
    RoundingMode);

template <typename T>
bool compare_ternary_operation_one_output(Operation op,
                                          const TernaryInput<T> &input,
                                          T libc_result, double ulp_tolerance,
                                          RoundingMode rounding) {
  unsigned int precision = get_precision<T>(ulp_tolerance);
  MPFRNumber mpfr_result = ternary_operation_one_output(
      op, input.x, input.y, input.z, precision, rounding);
  double ulp = mpfr_result.ulp(libc_result);

  return (ulp <= ulp_tolerance);
}

template bool compare_ternary_operation_one_output<float>(
    Operation, const TernaryInput<float> &, float, double, RoundingMode);
template bool compare_ternary_operation_one_output<double>(
    Operation, const TernaryInput<double> &, double, double, RoundingMode);
template bool compare_ternary_operation_one_output<long double>(
    Operation, const TernaryInput<long double> &, long double, double,
    RoundingMode);

} // namespace internal

template <typename T> bool round_to_long(T x, long &result) {
  MPFRNumber mpfr(x);
  return mpfr.round_to_long(result);
}

template bool round_to_long<float>(float, long &);
template bool round_to_long<double>(double, long &);
template bool round_to_long<long double>(long double, long &);

template <typename T> bool round_to_long(T x, RoundingMode mode, long &result) {
  MPFRNumber mpfr(x);
  return mpfr.round_to_long(get_mpfr_rounding_mode(mode), result);
}

template bool round_to_long<float>(float, RoundingMode, long &);
template bool round_to_long<double>(double, RoundingMode, long &);
template bool round_to_long<long double>(long double, RoundingMode, long &);

template <typename T> T round(T x, RoundingMode mode) {
  MPFRNumber mpfr(x);
  MPFRNumber result = mpfr.rint(get_mpfr_rounding_mode(mode));
  return result.as<T>();
}

template float round<float>(float, RoundingMode);
template double round<double>(double, RoundingMode);
template long double round<long double>(long double, RoundingMode);

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc
