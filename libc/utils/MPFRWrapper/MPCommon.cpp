//===-- Utils used by both MPCWrapper and MPFRWrapper----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPCommon.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {
namespace mpfr {

MPFRNumber::MPFRNumber() : mpfr_precision(256), mpfr_rounding(MPFR_RNDN) {
  mpfr_init2(value, mpfr_precision);
}

MPFRNumber::MPFRNumber(const MPFRNumber &other)
    : mpfr_precision(other.mpfr_precision), mpfr_rounding(other.mpfr_rounding) {
  mpfr_init2(value, mpfr_precision);
  mpfr_set(value, other.value, mpfr_rounding);
}

MPFRNumber::MPFRNumber(const MPFRNumber &other, unsigned int precision)
    : mpfr_precision(precision), mpfr_rounding(other.mpfr_rounding) {
  mpfr_init2(value, mpfr_precision);
  mpfr_set(value, other.value, mpfr_rounding);
}

MPFRNumber::MPFRNumber(const mpfr_t x, unsigned int precision,
                       RoundingMode rounding)
    : mpfr_precision(precision),
      mpfr_rounding(get_mpfr_rounding_mode(rounding)) {
  mpfr_init2(value, mpfr_precision);
  mpfr_set(value, x, mpfr_rounding);
}

MPFRNumber::~MPFRNumber() { mpfr_clear(value); }

MPFRNumber &MPFRNumber::operator=(const MPFRNumber &rhs) {
  mpfr_precision = rhs.mpfr_precision;
  mpfr_rounding = rhs.mpfr_rounding;
  mpfr_set(value, rhs.value, mpfr_rounding);
  return *this;
}

bool MPFRNumber::is_nan() const { return mpfr_nan_p(value); }

MPFRNumber MPFRNumber::abs() const {
  MPFRNumber result(*this);
  mpfr_abs(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::acos() const {
  MPFRNumber result(*this);
  mpfr_acos(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::acosh() const {
  MPFRNumber result(*this);
  mpfr_acosh(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::add(const MPFRNumber &b) const {
  MPFRNumber result(*this);
  mpfr_add(result.value, value, b.value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::asin() const {
  MPFRNumber result(*this);
  mpfr_asin(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::asinh() const {
  MPFRNumber result(*this);
  mpfr_asinh(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::atan() const {
  MPFRNumber result(*this);
  mpfr_atan(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::atan2(const MPFRNumber &b) {
  MPFRNumber result(*this);
  mpfr_atan2(result.value, value, b.value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::atanh() const {
  MPFRNumber result(*this);
  mpfr_atanh(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::cbrt() const {
  MPFRNumber result(*this);
  mpfr_cbrt(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::ceil() const {
  MPFRNumber result(*this);
  mpfr_ceil(result.value, value);
  return result;
}

MPFRNumber MPFRNumber::cos() const {
  MPFRNumber result(*this);
  mpfr_cos(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::cosh() const {
  MPFRNumber result(*this);
  mpfr_cosh(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::cospi() const {
  MPFRNumber result(*this);

#if MPFR_VERSION_MAJOR > 4 ||                                                  \
    (MPFR_VERSION_MAJOR == 4 && MPFR_VERSION_MINOR >= 2)
  mpfr_cospi(result.value, value, mpfr_rounding);
  return result;
#else
  if (mpfr_integer_p(value)) {
    mpz_t integer;
    mpz_init(integer);
    mpfr_get_z(integer, value, mpfr_rounding);

    int d = mpz_tstbit(integer, 0);
    mpfr_set_si(result.value, d ? -1 : 1, mpfr_rounding);
    mpz_clear(integer);
    return result;
  }

  MPFRNumber value_pi(0.0, 1280);
  mpfr_const_pi(value_pi.value, MPFR_RNDN);
  mpfr_mul(value_pi.value, value_pi.value, value, MPFR_RNDN);
  mpfr_cos(result.value, value_pi.value, mpfr_rounding);

  return result;
#endif
}

MPFRNumber MPFRNumber::erf() const {
  MPFRNumber result(*this);
  mpfr_erf(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::exp() const {
  MPFRNumber result(*this);
  mpfr_exp(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::exp2() const {
  MPFRNumber result(*this);
  mpfr_exp2(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::exp2m1() const {
  // TODO: Only use mpfr_exp2m1 once CI and buildbots get MPFR >= 4.2.0.
#if MPFR_VERSION_MAJOR > 4 ||                                                  \
    (MPFR_VERSION_MAJOR == 4 && MPFR_VERSION_MINOR >= 2)
  MPFRNumber result(*this);
  mpfr_exp2m1(result.value, value, mpfr_rounding);
  return result;
#else
  unsigned int prec = mpfr_precision * 3;
  MPFRNumber result(*this, prec);

  float f = mpfr_get_flt(abs().value, mpfr_rounding);
  if (f > 0.5f && f < 0x1.0p30f) {
    mpfr_exp2(result.value, value, mpfr_rounding);
    mpfr_sub_ui(result.value, result.value, 1, mpfr_rounding);
    return result;
  }

  MPFRNumber ln2(2.0f, prec);
  // log(2)
  mpfr_log(ln2.value, ln2.value, mpfr_rounding);
  // x * log(2)
  mpfr_mul(result.value, value, ln2.value, mpfr_rounding);
  // e^(x * log(2)) - 1
  int ex = mpfr_expm1(result.value, result.value, mpfr_rounding);
  mpfr_subnormalize(result.value, ex, mpfr_rounding);
  return result;
#endif
}

MPFRNumber MPFRNumber::exp10() const {
  MPFRNumber result(*this);
  mpfr_exp10(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::exp10m1() const {
  // TODO: Only use mpfr_exp10m1 once CI and buildbots get MPFR >= 4.2.0.
#if MPFR_VERSION_MAJOR > 4 ||                                                  \
    (MPFR_VERSION_MAJOR == 4 && MPFR_VERSION_MINOR >= 2)
  MPFRNumber result(*this);
  mpfr_exp10m1(result.value, value, mpfr_rounding);
  return result;
#else
  unsigned int prec = mpfr_precision * 3;
  MPFRNumber result(*this, prec);

  MPFRNumber ln10(10.0f, prec);
  // log(10)
  mpfr_log(ln10.value, ln10.value, mpfr_rounding);
  // x * log(10)
  mpfr_mul(result.value, value, ln10.value, mpfr_rounding);
  // e^(x * log(10)) - 1
  int ex = mpfr_expm1(result.value, result.value, mpfr_rounding);
  mpfr_subnormalize(result.value, ex, mpfr_rounding);
  return result;
#endif
}

MPFRNumber MPFRNumber::expm1() const {
  MPFRNumber result(*this);
  mpfr_expm1(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::div(const MPFRNumber &b) const {
  MPFRNumber result(*this);
  mpfr_div(result.value, value, b.value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::floor() const {
  MPFRNumber result(*this);
  mpfr_floor(result.value, value);
  return result;
}

MPFRNumber MPFRNumber::fmod(const MPFRNumber &b) {
  MPFRNumber result(*this);
  mpfr_fmod(result.value, value, b.value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::frexp(int &exp) {
  MPFRNumber result(*this);
  mpfr_exp_t resultExp;
  mpfr_frexp(&resultExp, result.value, value, mpfr_rounding);
  exp = resultExp;
  return result;
}

MPFRNumber MPFRNumber::hypot(const MPFRNumber &b) {
  MPFRNumber result(*this);
  mpfr_hypot(result.value, value, b.value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::log() const {
  MPFRNumber result(*this);
  mpfr_log(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::log2() const {
  MPFRNumber result(*this);
  mpfr_log2(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::log10() const {
  MPFRNumber result(*this);
  mpfr_log10(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::log1p() const {
  MPFRNumber result(*this);
  mpfr_log1p(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::pow(const MPFRNumber &b) {
  MPFRNumber result(*this);
  mpfr_pow(result.value, value, b.value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::remquo(const MPFRNumber &divisor, int &quotient) {
  MPFRNumber remainder(*this);
  long q;
  mpfr_remquo(remainder.value, &q, value, divisor.value, mpfr_rounding);
  quotient = q;
  return remainder;
}

MPFRNumber MPFRNumber::round() const {
  MPFRNumber result(*this);
  mpfr_round(result.value, value);
  return result;
}

MPFRNumber MPFRNumber::roundeven() const {
  MPFRNumber result(*this);
#if MPFR_VERSION_MAJOR >= 4
  mpfr_roundeven(result.value, value);
#else
  mpfr_rint(result.value, value, MPFR_RNDN);
#endif
  return result;
}

bool MPFRNumber::round_to_long(long &result) const {
  // We first calculate the rounded value. This way, when converting
  // to long using mpfr_get_si, the rounding direction of MPFR_RNDN
  // (or any other rounding mode), does not have an influence.
  MPFRNumber roundedValue = round();
  mpfr_clear_erangeflag();
  result = mpfr_get_si(roundedValue.value, MPFR_RNDN);
  return mpfr_erangeflag_p();
}

bool MPFRNumber::round_to_long(mpfr_rnd_t rnd, long &result) const {
  MPFRNumber rint_result(*this);
  mpfr_rint(rint_result.value, value, rnd);
  return rint_result.round_to_long(result);
}

MPFRNumber MPFRNumber::rint(mpfr_rnd_t rnd) const {
  MPFRNumber result(*this);
  mpfr_rint(result.value, value, rnd);
  return result;
}

MPFRNumber MPFRNumber::mod_2pi() const {
  MPFRNumber result(0.0, 1280);
  MPFRNumber _2pi(0.0, 1280);
  mpfr_const_pi(_2pi.value, MPFR_RNDN);
  mpfr_mul_si(_2pi.value, _2pi.value, 2, MPFR_RNDN);
  mpfr_fmod(result.value, value, _2pi.value, MPFR_RNDN);
  return result;
}

MPFRNumber MPFRNumber::mod_pi_over_2() const {
  MPFRNumber result(0.0, 1280);
  MPFRNumber pi_over_2(0.0, 1280);
  mpfr_const_pi(pi_over_2.value, MPFR_RNDN);
  mpfr_mul_d(pi_over_2.value, pi_over_2.value, 0.5, MPFR_RNDN);
  mpfr_fmod(result.value, value, pi_over_2.value, MPFR_RNDN);
  return result;
}

MPFRNumber MPFRNumber::mod_pi_over_4() const {
  MPFRNumber result(0.0, 1280);
  MPFRNumber pi_over_4(0.0, 1280);
  mpfr_const_pi(pi_over_4.value, MPFR_RNDN);
  mpfr_mul_d(pi_over_4.value, pi_over_4.value, 0.25, MPFR_RNDN);
  mpfr_fmod(result.value, value, pi_over_4.value, MPFR_RNDN);
  return result;
}

MPFRNumber MPFRNumber::sin() const {
  MPFRNumber result(*this);
  mpfr_sin(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::sinpi() const {
  MPFRNumber result(*this);

#if MPFR_VERSION_MAJOR > 4 ||                                                  \
    (MPFR_VERSION_MAJOR == 4 && MPFR_VERSION_MINOR >= 2)

  mpfr_sinpi(result.value, value, mpfr_rounding);
  return result;
#else
  if (mpfr_integer_p(value)) {
    mpfr_set_si(result.value, 0, mpfr_rounding);
    return result;
  }

  MPFRNumber value_mul_two(*this);
  mpfr_mul_si(value_mul_two.value, value, 2, MPFR_RNDN);

  if (mpfr_integer_p(value_mul_two.value)) {
    auto d = mpfr_get_si(value, MPFR_RNDD);
    mpfr_set_si(result.value, (d & 1) ? -1 : 1, mpfr_rounding);
    return result;
  }

  MPFRNumber value_pi(0.0, 1280);
  mpfr_const_pi(value_pi.value, MPFR_RNDN);
  mpfr_mul(value_pi.value, value_pi.value, value, MPFR_RNDN);
  mpfr_sin(result.value, value_pi.value, mpfr_rounding);
  return result;
#endif
}

MPFRNumber MPFRNumber::sinh() const {
  MPFRNumber result(*this);
  mpfr_sinh(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::sqrt() const {
  MPFRNumber result(*this);
  mpfr_sqrt(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::sub(const MPFRNumber &b) const {
  MPFRNumber result(*this);
  mpfr_sub(result.value, value, b.value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::tan() const {
  MPFRNumber result(*this);
  mpfr_tan(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::tanh() const {
  MPFRNumber result(*this);
  mpfr_tanh(result.value, value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::tanpi() const {
  MPFRNumber result(*this);

#if MPFR_VERSION_MAJOR > 4 ||                                                  \
    (MPFR_VERSION_MAJOR == 4 && MPFR_VERSION_MINOR >= 2)

  mpfr_tanpi(result.value, value, mpfr_rounding);
  return result;
#else
  MPFRNumber value_ret_exact(*this);
  MPFRNumber value_one(*this);
  mpfr_set_si(value_one.value, 1, MPFR_RNDN);
  mpfr_fmod(value_ret_exact.value, value, value_one.value, mpfr_rounding);
  mpfr_mul_si(value_ret_exact.value, value_ret_exact.value, 4, MPFR_RNDN);

  if (mpfr_integer_p(value_ret_exact.value)) {
    int mod = mpfr_get_si(value_ret_exact.value, MPFR_RNDN);
    mod = (mod < 0 ? -1 * mod : mod);

    switch (mod) {
    case 0:
      mpfr_set_si(result.value, 0, mpfr_rounding);
      break;
    case 1:
      mpfr_set_si(result.value, (mpfr_signbit(value) ? -1 : 1), mpfr_rounding);
      break;
    case 2: {
      auto d = mpfr_get_si(value, MPFR_RNDZ);
      d += mpfr_sgn(value) > 0 ? 0 : 1;
      mpfr_set_inf(result.value, (d & 1) ? -1 : 1);
      break;
    }
    case 3:
      mpfr_set_si(result.value, (mpfr_signbit(value) ? 1 : -1), mpfr_rounding);
      break;
    }

    return result;
  }

  MPFRNumber value_pi(0.0, 1280);
  mpfr_const_pi(value_pi.value, MPFR_RNDN);
  mpfr_mul(value_pi.value, value_pi.value, value, MPFR_RNDN);
  mpfr_tan(result.value, value_pi.value, mpfr_rounding);
  return result;
#endif
}

MPFRNumber MPFRNumber::trunc() const {
  MPFRNumber result(*this);
  mpfr_trunc(result.value, value);
  return result;
}

MPFRNumber MPFRNumber::fma(const MPFRNumber &b, const MPFRNumber &c) {
  MPFRNumber result(*this);
  mpfr_fma(result.value, value, b.value, c.value, mpfr_rounding);
  return result;
}

MPFRNumber MPFRNumber::mul(const MPFRNumber &b) {
  MPFRNumber result(*this);
  mpfr_mul(result.value, value, b.value, mpfr_rounding);
  return result;
}

cpp::string MPFRNumber::str() const {
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
  return cpp::string(view.data());
}

void MPFRNumber::dump(const char *msg) const {
  mpfr_printf("%s%.128g\n", msg, value);
}

template <> float MPFRNumber::as<float>() const {
  return mpfr_get_flt(value, mpfr_rounding);
}

template <> double MPFRNumber::as<double>() const {
  return mpfr_get_d(value, mpfr_rounding);
}

template <> long double MPFRNumber::as<long double>() const {
  return mpfr_get_ld(value, mpfr_rounding);
}

#ifdef LIBC_TYPES_HAS_FLOAT16
template <> float16 MPFRNumber::as<float16>() const {
  // TODO: Either prove that this cast won't cause double-rounding errors, or
  // find a better way to get a float16.
  return fputil::cast<float16>(mpfr_get_d(value, mpfr_rounding));
}
#endif

#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
template <> float128 MPFRNumber::as<float128>() const {
  return mpfr_get_float128(value, mpfr_rounding);
}

#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE

} // namespace mpfr
} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
