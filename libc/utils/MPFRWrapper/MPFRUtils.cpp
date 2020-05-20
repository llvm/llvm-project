//===-- Utils which wrap MPFR ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPFRUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <mpfr.h>
#include <stdint.h>
#include <string>

namespace __llvm_libc {
namespace testing {
namespace mpfr {

template <typename T> struct FloatProperties {};

template <> struct FloatProperties<float> {
  typedef uint32_t BitsType;
  static_assert(sizeof(BitsType) == sizeof(float),
                "Unexpected size of 'float' type.");

  static constexpr uint32_t mantissaWidth = 23;
  static constexpr BitsType signMask = 0x7FFFFFFFU;
  static constexpr uint32_t exponentOffset = 127;
};

template <> struct FloatProperties<double> {
  typedef uint64_t BitsType;
  static_assert(sizeof(BitsType) == sizeof(double),
                "Unexpected size of 'double' type.");

  static constexpr uint32_t mantissaWidth = 52;
  static constexpr BitsType signMask = 0x7FFFFFFFFFFFFFFFULL;
  static constexpr uint32_t exponentOffset = 1023;
};

template <typename T> typename FloatProperties<T>::BitsType getBits(T x) {
  using BitsType = typename FloatProperties<T>::BitsType;
  return *reinterpret_cast<BitsType *>(&x);
}

// Returns the zero adjusted exponent value of abs(x).
template <typename T> int getExponent(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename Properties::BitsType;
  BitsType bits = *reinterpret_cast<BitsType *>(&x);
  bits &= Properties::signMask;                // Zero the sign bit.
  int e = (bits >> Properties::mantissaWidth); // Shift out the mantissa.
  e -= Properties::exponentOffset;             // Zero adjust.
  return e;
}

class MPFRNumber {
  // A precision value which allows sufficiently large additional
  // precision even compared to double precision floating point values.
  static constexpr unsigned int mpfrPrecision = 96;

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
            cpp::EnableIfType<cpp::IsIntegral<XType>::Value, int> = 0>
  explicit MPFRNumber(XType x) {
    mpfr_init2(value, mpfrPrecision);
    mpfr_set_sj(value, x, MPFR_RNDN);
  }

  template <typename XType> MPFRNumber(XType x, const Tolerance &t) {
    mpfr_init2(value, mpfrPrecision);
    mpfr_set_zero(value, 1); // Set to positive zero.
    MPFRNumber xExponent(getExponent(x));
    // E = 2^E
    mpfr_exp2(xExponent.value, xExponent.value, MPFR_RNDN);
    uint32_t bitMask = 1 << (t.width - 1);
    for (int n = -t.basePrecision; bitMask > 0; bitMask >>= 1) {
      --n;
      if (t.bits & bitMask) {
        // delta = -n
        MPFRNumber delta(n);

        // delta = 2^(-n)
        mpfr_exp2(delta.value, delta.value, MPFR_RNDN);

        // delta = E * 2^(-n)
        mpfr_mul(delta.value, delta.value, xExponent.value, MPFR_RNDN);

        // tolerance += delta
        mpfr_add(value, value, delta.value, MPFR_RNDN);
      }
    }
  }

  template <typename XType,
            cpp::EnableIfType<cpp::IsFloatingPointType<XType>::Value, int> = 0>
  MPFRNumber(Operation op, XType rawValue) {
    mpfr_init2(value, mpfrPrecision);
    MPFRNumber mpfrInput(rawValue);
    switch (op) {
    case OP_Cos:
      mpfr_cos(value, mpfrInput.value, MPFR_RNDN);
      break;
    case OP_Sin:
      mpfr_sin(value, mpfrInput.value, MPFR_RNDN);
      break;
    }
  }

  MPFRNumber(const MPFRNumber &other) {
    mpfr_set(value, other.value, MPFR_RNDN);
  }

  ~MPFRNumber() { mpfr_clear(value); }

  // Returns true if |other| is within the |tolerance| value of this
  // number.
  bool isEqual(const MPFRNumber &other, const MPFRNumber &tolerance) const {
    MPFRNumber difference;
    if (mpfr_cmp(value, other.value) >= 0)
      mpfr_sub(difference.value, value, other.value, MPFR_RNDN);
    else
      mpfr_sub(difference.value, other.value, value, MPFR_RNDN);

    return mpfr_lessequal_p(difference.value, tolerance.value);
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
  float asFloat() const { return mpfr_get_flt(value, MPFR_RNDN); }
  double asDouble() const { return mpfr_get_d(value, MPFR_RNDN); }
  void dump(const char *msg) const { mpfr_printf("%s%.128Rf\n", msg, value); }
};

namespace internal {

template <typename T>
void MPFRMatcher<T>::explainError(testutils::StreamWrapper &OS) {
  MPFRNumber mpfrResult(operation, input);
  MPFRNumber mpfrInput(input);
  MPFRNumber mpfrMatchValue(matchValue);
  MPFRNumber mpfrToleranceValue(matchValue, tolerance);
  OS << "Match value not within tolerance value of MPFR result:\n"
     << "  Input decimal: " << mpfrInput.str() << '\n'
     << "     Input bits: 0x" << llvm::utohexstr(getBits(input)) << '\n'
     << "  Match decimal: " << mpfrMatchValue.str() << '\n'
     << "     Match bits: 0x" << llvm::utohexstr(getBits(matchValue)) << '\n'
     << "    MPFR result: " << mpfrResult.str() << '\n'
     << "Tolerance value: " << mpfrToleranceValue.str() << '\n';
}

template void MPFRMatcher<float>::explainError(testutils::StreamWrapper &);
template void MPFRMatcher<double>::explainError(testutils::StreamWrapper &);

template <typename T>
bool compare(Operation op, T input, T libcResult, const Tolerance &t) {
  MPFRNumber mpfrResult(op, input);
  MPFRNumber mpfrLibcResult(libcResult);
  MPFRNumber mpfrToleranceValue(libcResult, t);

  return mpfrResult.isEqual(mpfrLibcResult, mpfrToleranceValue);
};

template bool compare<float>(Operation, float, float, const Tolerance &);
template bool compare<double>(Operation, double, double, const Tolerance &);

} // namespace internal

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc
