//===-- Utils which wrap MPFR ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPCUtils.h"

#include "../MPFRWrapper/MPFRUtils.cpp"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/fpbits_str.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

#include <stdint.h>

#include "mpc.h"

template <typename T> using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;

namespace LIBC_NAMESPACE_DECL {
namespace testing {
namespace mpc {

// A precision value which allows sufficiently large additional
// precision compared to the floating point precision.
template <typename T> struct ExtraPrecision;

template <> struct ExtraPrecision<float> {
  static constexpr unsigned int VALUE = 128;
};

template <> struct ExtraPrecision<double> {
  static constexpr unsigned int VALUE = 256;
};

// If the ulp tolerance is less than or equal to 0.5, we would check that the
// result is rounded correctly with respect to the rounding mode by using the
// same precision as the inputs.
template <typename T>
static inline unsigned int get_precision(double ulp_tolerance) {
  if (ulp_tolerance <= 0.5) {
    return LIBC_NAMESPACE::fputil::FPBits<T>::FRACTION_LEN + 1;
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
  __builtin_unreachable();
}

template <typename T> struct MPCComplex {
  T real;
  T imag;
};

class MPCNumber {
private:
  unsigned int mpc_real_precision;
  unsigned int mpc_imag_precision;
  mpc_t value;
  mpc_rnd_t mpc_rounding;

public:
  MPCNumber(unsigned int r_p, unsigned int i_p)
      : mpc_real_precision(r_p), mpc_imag_precision(i_p),
        mpc_rounding(MPC_RNDNN) {
    mpc_init3(value, mpc_real_precision, mpc_imag_precision);
  }

  MPCNumber()
      : mpc_real_precision(256), mpc_imag_precision(256),
        mpc_rounding(MPC_RNDNN) {
    mpc_init2(value, 256);
  }

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<_Complex float, XType>, bool> = 0>
  MPCNumber(XType x, unsigned int precision = ExtraPrecision<float>::VALUE,
            MPCRoundingMode rounding = MPCRoundingMode(RoundingMode::Nearest,
                                                       RoundingMode::Nearest))
      : mpc_real_precision(precision), mpc_imag_precision(precision),
        mpc_rounding(MPC_RND(get_mpfr_rounding_mode(rounding.Rrnd),
                             get_mpfr_rounding_mode(rounding.Irnd))) {
    mpc_init2(value, precision);
    MPCComplex<float> x_c = cpp::bit_cast<MPCComplex<float>>(x);
    mpfr_t real, imag;
    mpfr_init2(real, precision);
    mpfr_init2(imag, precision);
    mpfr_set_flt(real, x_c.real, get_mpfr_rounding_mode(rounding.Rrnd));
    mpfr_set_flt(imag, x_c.imag, get_mpfr_rounding_mode(rounding.Irnd));
    mpc_set_fr_fr(value, real, imag, mpc_rounding);
  }

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<_Complex double, XType>, bool> = 0>
  MPCNumber(XType x, unsigned int precision = ExtraPrecision<double>::VALUE,
            MPCRoundingMode rounding = MPCRoundingMode(RoundingMode::Nearest,
                                                       RoundingMode::Nearest))
      : mpc_real_precision(precision), mpc_imag_precision(precision),
        mpc_rounding(MPC_RND(get_mpfr_rounding_mode(rounding.Rrnd),
                             get_mpfr_rounding_mode(rounding.Irnd))) {
    mpc_init2(value, precision);
    MPCComplex<double> x_c = cpp::bit_cast<MPCComplex<double>>(x);
    mpc_set_d_d(value, x_c.real, x_c.imag, mpc_rounding);
  }

  MPCNumber(const MPCNumber &other)
      : mpc_real_precision(other.mpc_real_precision),
        mpc_imag_precision(other.mpc_imag_precision),
        mpc_rounding(other.mpc_rounding) {
    mpc_init3(value, mpc_real_precision, mpc_imag_precision);
    mpc_set(value, other.value, mpc_rounding);
  }

  MPCNumber &operator=(const MPCNumber &rhs) {
    mpc_real_precision = rhs.mpc_real_precision;
    mpc_imag_precision = rhs.mpc_imag_precision;
    mpc_rounding = rhs.mpc_rounding;
    mpc_set(value, rhs.value, mpc_rounding);
    return *this;
  }

  MPCNumber(const mpc_t x, unsigned int r_p, unsigned int i_p, mpc_rnd_t rnd)
      : mpc_real_precision(r_p), mpc_imag_precision(i_p), mpc_rounding(rnd) {
    mpc_init3(value, mpc_real_precision, mpc_imag_precision);
    mpc_set(value, x, mpc_rounding);
  }

  ~MPCNumber() { mpc_clear(value); }

  void getValue(mpc_t val) const { mpc_set(val, value, mpc_rounding); }

  MPCNumber carg() const {
    mpfr_t res;
    mpc_t res_mpc;

    mpfr_init2(res, this->mpc_real_precision);
    mpc_init3(res_mpc, this->mpc_real_precision, this->mpc_imag_precision);

    mpc_arg(res, value, MPC_RND_RE(this->mpc_rounding));
    mpc_set_fr(res_mpc, res, this->mpc_rounding);

    MPCNumber result(res_mpc, this->mpc_real_precision, this->mpc_imag_precision, this->mpc_rounding);

    mpfr_clear(res);
    mpc_clear(res_mpc);

    return result;
  }
};

namespace internal {

template <typename InputType>
cpp::enable_if_t<cpp::is_complex_v<InputType>, MPCNumber>
unary_operation(Operation op, InputType input, unsigned int precision,
                MPCRoundingMode rounding) {
  MPCNumber mpcInput(input, precision, rounding);
  switch (op) {
  case Operation::Carg:
    return mpcInput.carg();
  default:
    __builtin_unreachable();
  }
}

template <typename InputType, typename OutputType>
bool compare_unary_operation_single_output_same_type(Operation op,
                                                     InputType input,
                                                     OutputType libc_result,
                                                     double ulp_tolerance,
                                                     MPCRoundingMode rounding) {
  unsigned int precision = get_precision<get_real_t<InputType>>(ulp_tolerance);
  MPCNumber mpc_result;
  mpc_result = unary_operation(op, input, precision, rounding);
  mpfr_t real, imag;
  mpc_t mpc_result_val;
  mpc_result.getValue(mpc_result_val);
  mpc_real(real, mpc_result_val, get_mpfr_rounding_mode(rounding.Rrnd));
  mpc_imag(imag, mpc_result_val, get_mpfr_rounding_mode(rounding.Irnd));
  mpfr::MPFRNumber mpfr_real(real, precision, rounding.Rrnd);
  mpfr::MPFRNumber mpfr_imag(imag, precision, rounding.Irnd);
  double ulp_real = mpfr_real.ulp(
      (cpp::bit_cast<MPCComplex<get_real_t<InputType>>>(libc_result)).real);
  double ulp_imag = mpfr_imag.ulp(
      (cpp::bit_cast<MPCComplex<get_real_t<InputType>>>(libc_result)).imag);
  return ((ulp_real <= ulp_tolerance) && (ulp_imag <= ulp_tolerance));
}

template bool compare_unary_operation_single_output_same_type(
    Operation, _Complex float, _Complex float, double, MPCRoundingMode);
template bool compare_unary_operation_single_output_same_type(
    Operation, _Complex double, _Complex double, double, MPCRoundingMode);

template <typename InputType, typename OutputType>
bool compare_unary_operation_single_output_different_type(
    Operation op, InputType input, OutputType libc_result, double ulp_tolerance,
    MPCRoundingMode rounding) {
  unsigned int precision = get_precision<get_real_t<InputType>>(ulp_tolerance);
  MPCNumber mpc_result;
  mpc_result = unary_operation(op, input, precision, rounding);
  mpc_t mpc_result_val;
  mpc_init3(mpc_result_val, precision, precision);
  mpc_result.getValue(mpc_result_val);
  mpfr_t real;
  mpfr_init2(real, precision);
  mpc_real(real, mpc_result_val, get_mpfr_rounding_mode(rounding.Rrnd));
  mpfr::MPFRNumber mpfr_real(real, precision, rounding.Rrnd);
  double ulp_real = mpfr_real.ulp(libc_result);
  return (ulp_real <= ulp_tolerance);
}

template bool compare_unary_operation_single_output_different_type(
    Operation, _Complex float, float, double, MPCRoundingMode);
template bool compare_unary_operation_single_output_different_type(
    Operation, _Complex double, double, double, MPCRoundingMode);

template <typename InputType, typename OutputType>
void explain_unary_operation_single_output_different_type_error(
    Operation op, InputType input, OutputType libc_result, double ulp_tolerance,
    MPCRoundingMode rounding) {

  unsigned int precision = get_precision<get_real_t<InputType>>(ulp_tolerance);

  MPCNumber mpc_result;
  mpc_result = unary_operation(op, input, precision, rounding);

  mpc_t mpc_result_val;
  mpc_init3(mpc_result_val, precision, precision);
  mpc_result.getValue(mpc_result_val);

  mpfr_t real;
  mpfr_init2(real, precision);
  mpc_real(real, mpc_result_val, get_mpfr_rounding_mode(rounding.Rrnd));

  mpfr::MPFRNumber mpfr_real(real, precision, rounding.Rrnd);

  double ulp_real = mpfr_real.ulp(libc_result);

  if(ulp_real > ulp_tolerance) {
    cpp::array<char, 1024> msg_buf;
    cpp::StringStream msg(msg_buf);
    // TODO: Add information to the error message.
  }

}

template void explain_unary_operation_single_output_different_type_error(
    Operation, _Complex float, float, double, MPCRoundingMode);
template void explain_unary_operation_single_output_different_type_error(
    Operation, _Complex double, double, double, MPCRoundingMode);

} // namespace internal

} // namespace mpc
} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
