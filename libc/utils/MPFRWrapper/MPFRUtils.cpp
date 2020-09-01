//===-- Utils which wrap MPFR ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPFRUtils.h"

#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <mpfr.h>
#include <stdint.h>
#include <string>

template <typename T> using FPBits = __llvm_libc::fputil::FPBits<T>;

namespace __llvm_libc {
namespace testing {
namespace mpfr {

class MPFRNumber {
  // A precision value which allows sufficiently large additional
  // precision even compared to quad-precision floating point values.
  static constexpr unsigned int mpfrPrecision = 128;

  mpfr_t value;

public:
  MPFRNumber() { mpfr_init2(value, mpfrPrecision); }

  // We use explicit EnableIf specializations to disallow implicit
  // conversions. Implicit conversions can potentially lead to loss of
  // precision.
  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<float, XType>::Value, int> = 0>
  explicit MPFRNumber(XType x) {
    mpfr_init2(value, mpfrPrecision);
    mpfr_set_flt(value, x, MPFR_RNDN);
  }

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<double, XType>::Value, int> = 0>
  explicit MPFRNumber(XType x) {
    mpfr_init2(value, mpfrPrecision);
    mpfr_set_d(value, x, MPFR_RNDN);
  }

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<long double, XType>::Value, int> = 0>
  explicit MPFRNumber(XType x) {
    mpfr_init2(value, mpfrPrecision);
    mpfr_set_ld(value, x, MPFR_RNDN);
  }

  template <typename XType,
            cpp::EnableIfType<cpp::IsIntegral<XType>::Value, int> = 0>
  explicit MPFRNumber(XType x) {
    mpfr_init2(value, mpfrPrecision);
    mpfr_set_sj(value, x, MPFR_RNDN);
  }

  MPFRNumber(const MPFRNumber &other) {
    mpfr_set(value, other.value, MPFR_RNDN);
  }

  ~MPFRNumber() {
    mpfr_clear(value);
  }

  MPFRNumber &operator=(const MPFRNumber &rhs) {
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

  MPFRNumber trunc() const {
    MPFRNumber result;
    mpfr_trunc(result.value, value);
    return result;
  }

  std::string str() const {
    // 200 bytes should be more than sufficient to hold a 100-digit number
    // plus additional bytes for the decimal point, '-' sign etc.
    constexpr size_t printBufSize = 200;
    char buffer[printBufSize];
    mpfr_snprintf(buffer, printBufSize, "%100.50Rf", value);
    llvm::StringRef ref(buffer);
    ref = ref.trim();
    return ref.str();
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
  // We define:
  //   ULP(mpfr_value, value) = abs(mpfr_value - value) / eps(value)
  //
  // Remarks:
  // 1. ULP < 0.5 will imply that the value is correctly rounded.
  // 2. We expect that this value and the value to be compared (the [input]
  //    argument) are reasonable close, and we will provide an upper bound
  //    of ULP value for testing.  Morever, most of the fractional parts of
  //    ULP value do not matter much, so using double as the return type
  //    should be good enough.
  template <typename T>
  cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, double> ulp(T input) {
    fputil::FPBits<T> bits(input);
    MPFRNumber mpfrInput(input);

    // abs(value - input)
    mpfr_sub(mpfrInput.value, value, mpfrInput.value, MPFR_RNDN);
    mpfr_abs(mpfrInput.value, mpfrInput.value, MPFR_RNDN);

    // get eps(input)
    int epsExponent = bits.exponent - fputil::FPBits<T>::exponentBias -
                      fputil::MantissaWidth<T>::value;
    if (bits.exponent == 0) {
      // correcting denormal exponent
      ++epsExponent;
    } else if ((bits.mantissa == 0) && (bits.exponent > 1) &&
               mpfr_less_p(value, mpfrInput.value)) {
      // when the input is exactly 2^n, distance (epsilon) between the input
      // and the next floating point number is different from the distance to
      // the previous floating point number.  So in that case, if the correct
      // value from MPFR is smaller than the input, we use the smaller epsilon
      --epsExponent;
    }

    // Since eps(value) is of the form 2^e, instead of dividing such number,
    // we multiply by its inverse 2^{-e}.
    mpfr_mul_2si(mpfrInput.value, mpfrInput.value, -epsExponent, MPFR_RNDN);

    return mpfrInput.as<double>();
  }
};

namespace internal {

template <typename InputType>
cpp::EnableIfType<cpp::IsFloatingPointType<InputType>::Value, MPFRNumber>
unaryOperation(Operation op, InputType input) {
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
  case Operation::Floor:
    return mpfrInput.floor();
  case Operation::Round:
    return mpfrInput.round();
  case Operation::Sin:
    return mpfrInput.sin();
  case Operation::Sqrt:
    return mpfrInput.sqrt();
  case Operation::Trunc:
    return mpfrInput.trunc();
  default:
    __builtin_unreachable();
  }
}

template <typename InputType>
cpp::EnableIfType<cpp::IsFloatingPointType<InputType>::Value, MPFRNumber>
unaryOperationTwoOutputs(Operation op, InputType input, int &output) {
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
binaryOperationTwoOutputs(Operation op, InputType x, InputType y, int &output) {
  MPFRNumber inputX(x), inputY(y);
  switch (op) {
  case Operation::RemQuo:
    return inputX.remquo(inputY, output);
  default:
    __builtin_unreachable();
  }
}

template <typename T>
void explainUnaryOperationSingleOutputError(Operation op, T input, T matchValue,
                                            testutils::StreamWrapper &OS) {
  MPFRNumber mpfrInput(input);
  MPFRNumber mpfrResult = unaryOperation(op, input);
  MPFRNumber mpfrMatchValue(matchValue);
  FPBits<T> inputBits(input);
  FPBits<T> matchBits(matchValue);
  FPBits<T> mpfrResultBits(mpfrResult.as<T>());
  OS << "Match value not within tolerance value of MPFR result:\n"
     << "  Input decimal: " << mpfrInput.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("     Input bits: ", input, OS);
  OS << '\n' << "  Match decimal: " << mpfrMatchValue.str() << '\n';
  __llvm_libc::fputil::testing::describeValue("     Match bits: ", matchValue,
                                              OS);
  OS << '\n' << "    MPFR result: " << mpfrResult.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "   MPFR rounded: ", mpfrResult.as<T>(), OS);
  OS << '\n';
  OS << "      ULP error: " << std::to_string(mpfrResult.ulp(matchValue))
     << '\n';
}

template void
explainUnaryOperationSingleOutputError<float>(Operation op, float, float,
                                              testutils::StreamWrapper &);
template void
explainUnaryOperationSingleOutputError<double>(Operation op, double, double,
                                               testutils::StreamWrapper &);
template void explainUnaryOperationSingleOutputError<long double>(
    Operation op, long double, long double, testutils::StreamWrapper &);

template <typename T>
void explainUnaryOperationTwoOutputsError(Operation op, T input,
                                          const BinaryOutput<T> &libcResult,
                                          testutils::StreamWrapper &OS) {
  MPFRNumber mpfrInput(input);
  FPBits<T> inputBits(input);
  int mpfrIntResult;
  MPFRNumber mpfrResult = unaryOperationTwoOutputs(op, input, mpfrIntResult);

  if (mpfrIntResult != libcResult.i) {
    OS << "MPFR integral result: " << mpfrIntResult << '\n'
       << "Libc integral result: " << libcResult.i << '\n';
  } else {
    OS << "Integral result from libc matches integral result from MPFR.\n";
  }

  MPFRNumber mpfrMatchValue(libcResult.f);
  OS << "Libc floating point result is not within tolerance value of the MPFR "
     << "result.\n\n";

  OS << "            Input decimal: " << mpfrInput.str() << "\n\n";

  OS << "Libc floating point value: " << mpfrMatchValue.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      " Libc floating point bits: ", libcResult.f, OS);
  OS << "\n\n";

  OS << "              MPFR result: " << mpfrResult.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "             MPFR rounded: ", mpfrResult.as<T>(), OS);
  OS << '\n'
     << "                ULP error: "
     << std::to_string(mpfrResult.ulp(libcResult.f)) << '\n';
}

template void explainUnaryOperationTwoOutputsError<float>(
    Operation, float, const BinaryOutput<float> &, testutils::StreamWrapper &);
template void
explainUnaryOperationTwoOutputsError<double>(Operation, double,
                                             const BinaryOutput<double> &,
                                             testutils::StreamWrapper &);
template void explainUnaryOperationTwoOutputsError<long double>(
    Operation, long double, const BinaryOutput<long double> &,
    testutils::StreamWrapper &);

template <typename T>
void explainBinaryOperationTwoOutputsError(Operation op,
                                           const BinaryInput<T> &input,
                                           const BinaryOutput<T> &libcResult,
                                           testutils::StreamWrapper &OS) {
  MPFRNumber mpfrX(input.x);
  MPFRNumber mpfrY(input.y);
  FPBits<T> xbits(input.x);
  FPBits<T> ybits(input.y);
  int mpfrIntResult;
  MPFRNumber mpfrResult =
      binaryOperationTwoOutputs(op, input.x, input.y, mpfrIntResult);
  MPFRNumber mpfrMatchValue(libcResult.f);

  OS << "Input decimal: x: " << mpfrX.str() << " y: " << mpfrY.str() << '\n'
     << "MPFR integral result: " << mpfrIntResult << '\n'
     << "Libc integral result: " << libcResult.i << '\n'
     << "Libc floating point result: " << mpfrMatchValue.str() << '\n'
     << "               MPFR result: " << mpfrResult.str() << '\n';
  __llvm_libc::fputil::testing::describeValue(
      "Libc floating point result bits: ", libcResult.f, OS);
  __llvm_libc::fputil::testing::describeValue(
      "              MPFR rounded bits: ", mpfrResult.as<T>(), OS);
  OS << "ULP error: " << std::to_string(mpfrResult.ulp(libcResult.f)) << '\n';
}

template void explainBinaryOperationTwoOutputsError<float>(
    Operation, const BinaryInput<float> &, const BinaryOutput<float> &,
    testutils::StreamWrapper &);
template void explainBinaryOperationTwoOutputsError<double>(
    Operation, const BinaryInput<double> &, const BinaryOutput<double> &,
    testutils::StreamWrapper &);
template void explainBinaryOperationTwoOutputsError<long double>(
    Operation, const BinaryInput<long double> &,
    const BinaryOutput<long double> &, testutils::StreamWrapper &);

template <typename T>
bool compareUnaryOperationSingleOutput(Operation op, T input, T libcResult,
                                       double ulpError) {
  // If the ulp error is exactly 0.5 (i.e a tie), we would check that the result
  // is rounded to the nearest even.
  MPFRNumber mpfrResult = unaryOperation(op, input);
  double ulp = mpfrResult.ulp(libcResult);
  bool bitsAreEven = ((FPBits<T>(libcResult).bitsAsUInt() & 1) == 0);
  return (ulp < ulpError) ||
         ((ulp == ulpError) && ((ulp != 0.5) || bitsAreEven));
}

template bool compareUnaryOperationSingleOutput<float>(Operation, float, float,
                                                       double);
template bool compareUnaryOperationSingleOutput<double>(Operation, double,
                                                        double, double);
template bool compareUnaryOperationSingleOutput<long double>(Operation,
                                                             long double,
                                                             long double,
                                                             double);

template <typename T>
bool compareUnaryOperationTwoOutputs(Operation op, T input,
                                     const BinaryOutput<T> &libcResult,
                                     double ulpError) {
  int mpfrIntResult;
  MPFRNumber mpfrResult = unaryOperationTwoOutputs(op, input, mpfrIntResult);
  double ulp = mpfrResult.ulp(libcResult.f);

  if (mpfrIntResult != libcResult.i)
    return false;

  bool bitsAreEven = ((FPBits<T>(libcResult.f).bitsAsUInt() & 1) == 0);
  return (ulp < ulpError) ||
         ((ulp == ulpError) && ((ulp != 0.5) || bitsAreEven));
}

template bool
compareUnaryOperationTwoOutputs<float>(Operation, float,
                                       const BinaryOutput<float> &, double);
template bool
compareUnaryOperationTwoOutputs<double>(Operation, double,
                                        const BinaryOutput<double> &, double);
template bool compareUnaryOperationTwoOutputs<long double>(
    Operation, long double, const BinaryOutput<long double> &, double);

template <typename T>
bool compareBinaryOperationTwoOutputs(Operation op, const BinaryInput<T> &input,
                                      const BinaryOutput<T> &libcResult,
                                      double ulpError) {
  int mpfrIntResult;
  MPFRNumber mpfrResult =
      binaryOperationTwoOutputs(op, input.x, input.y, mpfrIntResult);
  double ulp = mpfrResult.ulp(libcResult.f);

  if (mpfrIntResult != libcResult.i) {
    if (op == Operation::RemQuo) {
      if ((0x7 & mpfrIntResult) != (0x7 & libcResult.i))
        return false;
    } else {
      return false;
    }
  }

  bool bitsAreEven = ((FPBits<T>(libcResult.f).bitsAsUInt() & 1) == 0);
  return (ulp < ulpError) ||
         ((ulp == ulpError) && ((ulp != 0.5) || bitsAreEven));
}

template bool
compareBinaryOperationTwoOutputs<float>(Operation, const BinaryInput<float> &,
                                        const BinaryOutput<float> &, double);
template bool
compareBinaryOperationTwoOutputs<double>(Operation, const BinaryInput<double> &,
                                         const BinaryOutput<double> &, double);
template bool compareBinaryOperationTwoOutputs<long double>(
    Operation, const BinaryInput<long double> &,
    const BinaryOutput<long double> &, double);

} // namespace internal

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc
