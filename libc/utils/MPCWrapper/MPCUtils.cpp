//===-- Utils which wrap MPC ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPCUtils.h"

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/stringstream.h"
#include "utils/MPFRWrapper/MPCommon.h"

#include <stdint.h>

#include "mpc.h"

template <typename T> using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;

namespace LIBC_NAMESPACE_DECL {
namespace testing {
namespace mpc {

static inline cpp::string str(RoundingMode mode) {
  switch (mode) {
  case RoundingMode::Upward:
    return "MPFR_RNDU";
  case RoundingMode::Downward:
    return "MPFR_RNDD";
  case RoundingMode::TowardZero:
    return "MPFR_RNDZ";
  case RoundingMode::Nearest:
    return "MPFR_RNDN";
  }
}

class MPCNumber {
private:
  unsigned int precision;
  mpc_t value;
  mpc_rnd_t mpc_rounding;

public:
  explicit MPCNumber(unsigned int p) : precision(p), mpc_rounding(MPC_RNDNN) {
    mpc_init2(value, precision);
  }

  MPCNumber() : precision(256), mpc_rounding(MPC_RNDNN) {
    mpc_init2(value, 256);
  }

  MPCNumber(unsigned int p, mpc_rnd_t rnd) : precision(p), mpc_rounding(rnd) {
    mpc_init2(value, precision);
  }

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<_Complex float, XType>, bool> = 0>
  MPCNumber(XType x,
            unsigned int precision = mpfr::ExtraPrecision<float>::VALUE,
            RoundingMode rnd = RoundingMode::Nearest)
      : precision(precision),
        mpc_rounding(MPC_RND(mpfr::get_mpfr_rounding_mode(rnd),
                             mpfr::get_mpfr_rounding_mode(rnd))) {
    mpc_init2(value, precision);
    Complex<float> x_c = cpp::bit_cast<Complex<float>>(x);
    mpfr_t real, imag;
    mpfr_init2(real, precision);
    mpfr_init2(imag, precision);
    mpfr_set_flt(real, x_c.real, mpfr::get_mpfr_rounding_mode(rnd));
    mpfr_set_flt(imag, x_c.imag, mpfr::get_mpfr_rounding_mode(rnd));
    mpc_set_fr_fr(value, real, imag, mpc_rounding);
    mpfr_clear(real);
    mpfr_clear(imag);
  }

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<_Complex double, XType>, bool> = 0>
  MPCNumber(XType x,
            unsigned int precision = mpfr::ExtraPrecision<double>::VALUE,
            RoundingMode rnd = RoundingMode::Nearest)
      : precision(precision),
        mpc_rounding(MPC_RND(mpfr::get_mpfr_rounding_mode(rnd),
                             mpfr::get_mpfr_rounding_mode(rnd))) {
    mpc_init2(value, precision);
    Complex<double> x_c = cpp::bit_cast<Complex<double>>(x);
    mpc_set_d_d(value, x_c.real, x_c.imag, mpc_rounding);
  }

  MPCNumber(const MPCNumber &other)
      : precision(other.precision), mpc_rounding(other.mpc_rounding) {
    mpc_init2(value, precision);
    mpc_set(value, other.value, mpc_rounding);
  }

  ~MPCNumber() { mpc_clear(value); }

  MPCNumber &operator=(const MPCNumber &rhs) {
    precision = rhs.precision;
    mpc_rounding = rhs.mpc_rounding;
    mpc_init2(value, precision);
    mpc_set(value, rhs.value, mpc_rounding);
    return *this;
  }

  void setValue(mpc_t val) const { mpc_set(val, value, mpc_rounding); }

  mpc_t &getValue() { return value; }

  MPCNumber carg() const {
    mpfr_t res;
    MPCNumber result(precision, mpc_rounding);

    mpfr_init2(res, precision);

    mpc_arg(res, value, MPC_RND_RE(mpc_rounding));
    mpc_set_fr(result.value, res, mpc_rounding);

    mpfr_clear(res);

    return result;
  }

  MPCNumber cproj() const {
    MPCNumber result(precision, mpc_rounding);
    mpc_proj(result.value, value, mpc_rounding);
    return result;
  }
};

namespace internal {

template <typename InputType>
cpp::enable_if_t<cpp::is_complex_v<InputType>, MPCNumber>
unary_operation(Operation op, InputType input, unsigned int precision,
                RoundingMode rounding) {
  MPCNumber mpcInput(input, precision, rounding);
  switch (op) {
  case Operation::Carg:
    return mpcInput.carg();
  case Operation::Cproj:
    return mpcInput.cproj();
  default:
    __builtin_unreachable();
  }
}

template <typename InputType, typename OutputType>
bool compare_unary_operation_single_output_same_type(Operation op,
                                                     InputType input,
                                                     OutputType libc_result,
                                                     double ulp_tolerance,
                                                     RoundingMode rounding) {

  unsigned int precision =
      mpfr::get_precision<make_real_t<InputType>>(ulp_tolerance);

  MPCNumber mpc_result;
  mpc_result = unary_operation(op, input, precision, rounding);

  mpc_t mpc_result_val;
  mpc_init2(mpc_result_val, precision);
  mpc_result.setValue(mpc_result_val);

  mpfr_t real, imag;
  mpfr_init2(real, precision);
  mpfr_init2(imag, precision);
  mpc_real(real, mpc_result_val, mpfr::get_mpfr_rounding_mode(rounding));
  mpc_imag(imag, mpc_result_val, mpfr::get_mpfr_rounding_mode(rounding));

  mpfr::MPFRNumber mpfr_real(real, precision, rounding);
  mpfr::MPFRNumber mpfr_imag(imag, precision, rounding);

  double ulp_real = mpfr_real.ulp(
      (cpp::bit_cast<Complex<make_real_t<InputType>>>(libc_result)).real);
  double ulp_imag = mpfr_imag.ulp(
      (cpp::bit_cast<Complex<make_real_t<InputType>>>(libc_result)).imag);
  mpc_clear(mpc_result_val);
  mpfr_clear(real);
  mpfr_clear(imag);
  return (ulp_real <= ulp_tolerance) && (ulp_imag <= ulp_tolerance);
}

template bool compare_unary_operation_single_output_same_type(
    Operation, _Complex float, _Complex float, double, RoundingMode);
template bool compare_unary_operation_single_output_same_type(
    Operation, _Complex double, _Complex double, double, RoundingMode);

template <typename InputType, typename OutputType>
bool compare_unary_operation_single_output_different_type(
    Operation op, InputType input, OutputType libc_result, double ulp_tolerance,
    RoundingMode rounding) {

  unsigned int precision =
      mpfr::get_precision<make_real_t<InputType>>(ulp_tolerance);

  MPCNumber mpc_result;
  mpc_result = unary_operation(op, input, precision, rounding);

  mpc_t mpc_result_val;
  mpc_init2(mpc_result_val, precision);
  mpc_result.setValue(mpc_result_val);

  mpfr_t real;
  mpfr_init2(real, precision);
  mpc_real(real, mpc_result_val, mpfr::get_mpfr_rounding_mode(rounding));

  mpfr::MPFRNumber mpfr_real(real, precision, rounding);

  double ulp_real = mpfr_real.ulp(libc_result);
  mpc_clear(mpc_result_val);
  mpfr_clear(real);
  return (ulp_real <= ulp_tolerance);
}

template bool compare_unary_operation_single_output_different_type(
    Operation, _Complex float, float, double, RoundingMode);
template bool compare_unary_operation_single_output_different_type(
    Operation, _Complex double, double, double, RoundingMode);

template <typename InputType, typename OutputType>
void explain_unary_operation_single_output_different_type_error(
    Operation op, InputType input, OutputType libc_result, double ulp_tolerance,
    RoundingMode rounding) {

  unsigned int precision =
      mpfr::get_precision<make_real_t<InputType>>(ulp_tolerance);

  MPCNumber mpc_result;
  mpc_result = unary_operation(op, input, precision, rounding);

  mpc_t mpc_result_val;
  mpc_init2(mpc_result_val, precision);
  mpc_result.setValue(mpc_result_val);

  mpfr_t real;
  mpfr_init2(real, precision);
  mpc_real(real, mpc_result_val, mpfr::get_mpfr_rounding_mode(rounding));

  mpfr::MPFRNumber mpfr_result(real, precision, rounding);
  mpfr::MPFRNumber mpfrLibcResult(libc_result, precision, rounding);
  mpfr::MPFRNumber mpfrInputReal(
      cpp::bit_cast<Complex<make_real_t<InputType>>>(input).real, precision,
      rounding);
  mpfr::MPFRNumber mpfrInputImag(
      cpp::bit_cast<Complex<make_real_t<InputType>>>(input).imag, precision,
      rounding);

  cpp::array<char, 2048> msg_buf;
  cpp::StringStream msg(msg_buf);
  msg << "Match value not within tolerance value of MPFR result:\n"
      << "  Input: " << mpfrInputReal.str() << " + " << mpfrInputImag.str()
      << "i\n"
      << "  Rounding mode: " << str(rounding) << '\n'
      << "    Libc: " << mpfrLibcResult.str() << '\n'
      << "    MPC: " << mpfr_result.str() << '\n'
      << '\n'
      << "  ULP error: " << mpfr_result.ulp_as_mpfr_number(libc_result).str()
      << '\n';
  tlog << msg.str();
  mpc_clear(mpc_result_val);
  mpfr_clear(real);
}

template void explain_unary_operation_single_output_different_type_error(
    Operation, _Complex float, float, double, RoundingMode);
template void explain_unary_operation_single_output_different_type_error(
    Operation, _Complex double, double, double, RoundingMode);

template <typename InputType, typename OutputType>
void explain_unary_operation_single_output_same_type_error(
    Operation op, InputType input, OutputType libc_result, double ulp_tolerance,
    RoundingMode rounding) {

  unsigned int precision =
      mpfr::get_precision<make_real_t<InputType>>(ulp_tolerance);

  MPCNumber mpc_result;
  mpc_result = unary_operation(op, input, precision, rounding);

  mpc_t mpc_result_val;
  mpc_init2(mpc_result_val, precision);
  mpc_result.setValue(mpc_result_val);

  mpfr_t real, imag;
  mpfr_init2(real, precision);
  mpfr_init2(imag, precision);
  mpc_real(real, mpc_result_val, mpfr::get_mpfr_rounding_mode(rounding));
  mpc_imag(imag, mpc_result_val, mpfr::get_mpfr_rounding_mode(rounding));

  mpfr::MPFRNumber mpfr_real(real, precision, rounding);
  mpfr::MPFRNumber mpfr_imag(imag, precision, rounding);
  mpfr::MPFRNumber mpfrLibcResultReal(
      cpp::bit_cast<Complex<make_real_t<InputType>>>(libc_result).real,
      precision, rounding);
  mpfr::MPFRNumber mpfrLibcResultImag(
      cpp::bit_cast<Complex<make_real_t<InputType>>>(libc_result).imag,
      precision, rounding);
  mpfr::MPFRNumber mpfrInputReal(
      cpp::bit_cast<Complex<make_real_t<InputType>>>(input).real, precision,
      rounding);
  mpfr::MPFRNumber mpfrInputImag(
      cpp::bit_cast<Complex<make_real_t<InputType>>>(input).imag, precision,
      rounding);

  cpp::array<char, 2048> msg_buf;
  cpp::StringStream msg(msg_buf);
  msg << "Match value not within tolerance value of MPFR result:\n"
      << "  Input: " << mpfrInputReal.str() << " + " << mpfrInputImag.str()
      << "i\n"
      << "  Rounding mode: " << str(rounding) << " , " << str(rounding) << '\n'
      << "    Libc: " << mpfrLibcResultReal.str() << " + "
      << mpfrLibcResultImag.str() << "i\n"
      << "    MPC: " << mpfr_real.str() << " + " << mpfr_imag.str() << "i\n"
      << '\n'
      << "  ULP error: "
      << mpfr_real
             .ulp_as_mpfr_number(
                 cpp::bit_cast<Complex<make_real_t<InputType>>>(libc_result)
                     .real)
             .str()
      << " , "
      << mpfr_imag
             .ulp_as_mpfr_number(
                 cpp::bit_cast<Complex<make_real_t<InputType>>>(libc_result)
                     .imag)
             .str()
      << '\n';
  tlog << msg.str();
  mpc_clear(mpc_result_val);
  mpfr_clear(real);
  mpfr_clear(imag);
}

template void explain_unary_operation_single_output_same_type_error(
    Operation, _Complex float, _Complex float, double, RoundingMode);
template void explain_unary_operation_single_output_same_type_error(
    Operation, _Complex double, _Complex double, double, RoundingMode);

} // namespace internal

} // namespace mpc
} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
