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

  template <typename XType,
            cpp::EnableIfType<cpp::IsFloatingPointType<XType>::Value, int> = 0>
  MPFRNumber(Operation op, XType rawValue) {
    mpfr_init2(value, mpfrPrecision);
    MPFRNumber mpfrInput(rawValue);
    switch (op) {
    case Operation::Abs:
      mpfr_abs(value, mpfrInput.value, MPFR_RNDN);
      break;
    case Operation::Ceil:
      mpfr_ceil(value, mpfrInput.value);
      break;
    case Operation::Cos:
      mpfr_cos(value, mpfrInput.value, MPFR_RNDN);
      break;
    case Operation::Exp:
      mpfr_exp(value, mpfrInput.value, MPFR_RNDN);
      break;
    case Operation::Exp2:
      mpfr_exp2(value, mpfrInput.value, MPFR_RNDN);
      break;
    case Operation::Floor:
      mpfr_floor(value, mpfrInput.value);
      break;
    case Operation::Round:
      mpfr_round(value, mpfrInput.value);
      break;
    case Operation::Sin:
      mpfr_sin(value, mpfrInput.value, MPFR_RNDN);
      break;
    case Operation::Sqrt:
      mpfr_sqrt(value, mpfrInput.value, MPFR_RNDN);
      break;
    case Operation::Trunc:
      mpfr_trunc(value, mpfrInput.value);
      break;
    }
  }

  MPFRNumber(const MPFRNumber &other) {
    mpfr_set(value, other.value, MPFR_RNDN);
  }

  ~MPFRNumber() { mpfr_clear(value); }

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

template <typename T>
void MPFRMatcher<T>::explainError(testutils::StreamWrapper &OS) {
  MPFRNumber mpfrResult(operation, input);
  MPFRNumber mpfrInput(input);
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

template void MPFRMatcher<float>::explainError(testutils::StreamWrapper &);
template void MPFRMatcher<double>::explainError(testutils::StreamWrapper &);
template void
MPFRMatcher<long double>::explainError(testutils::StreamWrapper &);

template <typename T>
bool compare(Operation op, T input, T libcResult, double ulpError) {
  // If the ulp error is exactly 0.5 (i.e a tie), we would check that the result
  // is rounded to the nearest even.
  MPFRNumber mpfrResult(op, input);
  double ulp = mpfrResult.ulp(libcResult);
  bool bitsAreEven = ((FPBits<T>(libcResult).bitsAsUInt() & 1) == 0);
  return (ulp < ulpError) ||
         ((ulp == ulpError) && ((ulp != 0.5) || bitsAreEven));
}

template bool compare<float>(Operation, float, float, double);
template bool compare<double>(Operation, double, double, double);
template bool compare<long double>(Operation, long double, long double, double);

} // namespace internal

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc
