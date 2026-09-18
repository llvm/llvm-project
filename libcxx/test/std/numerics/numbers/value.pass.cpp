//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cassert>
#include <numbers>

#include "test_macros.h"

constexpr bool tests() {
  assert(std::numbers::e == 0x1.5bf0a8b145769p+1);
  assert(std::numbers::e_v<double> == 0x1.5bf0a8b145769p+1);
  assert(std::numbers::e_v<float> == 0x1.5bf0a8p+1f);

  assert(std::numbers::log2e == 0x1.71547652b82fep+0);
  assert(std::numbers::log2e_v<double> == 0x1.71547652b82fep+0);
  assert(std::numbers::log2e_v<float> == 0x1.715476p+0f);

  assert(std::numbers::log10e == 0x1.bcb7b1526e50ep-2);
  assert(std::numbers::log10e_v<double> == 0x1.bcb7b1526e50ep-2);
  assert(std::numbers::log10e_v<float> == 0x1.bcb7b15p-2f);

  assert(std::numbers::pi == 0x1.921fb54442d18p+1);
  assert(std::numbers::pi_v<double> == 0x1.921fb54442d18p+1);
  assert(std::numbers::pi_v<float> == 0x1.921fb54p+1f);

  assert(std::numbers::inv_pi == 0x1.45f306dc9c883p-2);
  assert(std::numbers::inv_pi_v<double> == 0x1.45f306dc9c883p-2);
  assert(std::numbers::inv_pi_v<float> == 0x1.45f306p-2f);

  assert(std::numbers::inv_sqrtpi == 0x1.20dd750429b6dp-1);
  assert(std::numbers::inv_sqrtpi_v<double> == 0x1.20dd750429b6dp-1);
  assert(std::numbers::inv_sqrtpi_v<float> == 0x1.20dd76p-1f);

  assert(std::numbers::ln2 == 0x1.62e42fefa39efp-1);
  assert(std::numbers::ln2_v<double> == 0x1.62e42fefa39efp-1);
  assert(std::numbers::ln2_v<float> == 0x1.62e42fp-1f);

  assert(std::numbers::ln10 == 0x1.26bb1bbb55516p+1);
  assert(std::numbers::ln10_v<double> == 0x1.26bb1bbb55516p+1);
  assert(std::numbers::ln10_v<float> == 0x1.26bb1bp+1f);

  assert(std::numbers::sqrt2 == 0x1.6a09e667f3bcdp+0);
  assert(std::numbers::sqrt2_v<double> == 0x1.6a09e667f3bcdp+0);
  assert(std::numbers::sqrt2_v<float> == 0x1.6a09e6p+0f);

  assert(std::numbers::sqrt3 == 0x1.bb67ae8584caap+0);
  assert(std::numbers::sqrt3_v<double> == 0x1.bb67ae8584caap+0);
  assert(std::numbers::sqrt3_v<float> == 0x1.bb67aep+0f);

  assert(std::numbers::inv_sqrt3 == 0x1.279a74590331cp-1);
  assert(std::numbers::inv_sqrt3_v<double> == 0x1.279a74590331cp-1);
  assert(std::numbers::inv_sqrt3_v<float> == 0x1.279a74p-1f);

  assert(std::numbers::egamma == 0x1.2788cfc6fb619p-1);
  assert(std::numbers::egamma_v<double> == 0x1.2788cfc6fb619p-1);
  assert(std::numbers::egamma_v<float> == 0x1.2788cfp-1f);

  assert(std::numbers::phi == 0x1.9e3779b97f4a8p+0);
  assert(std::numbers::phi_v<double> == 0x1.9e3779b97f4a8p+0);
  assert(std::numbers::phi_v<float> == 0x1.9e3779ap+0f);

#if defined(TEST_LONG_DOUBLE_IS_DOUBLE)
  assert(std::numbers::e_v<long double> == 0x1.5bf0a8b145769p+1L);
  assert(std::numbers::log2e_v<long double> == 0x1.71547652b82fep+0L);
  assert(std::numbers::log10e_v<long double> == 0x1.bcb7b1526e50ep-2L);
  assert(std::numbers::pi_v<long double> == 0x1.921fb54442d18p+1L);
  assert(std::numbers::inv_pi_v<long double> == 0x1.45f306dc9c883p-2L);
  assert(std::numbers::inv_sqrtpi_v<long double> == 0x1.20dd750429b6dp-1L);
  assert(std::numbers::ln2_v<long double> == 0x1.62e42fefa39efp-1L);
  assert(std::numbers::ln10_v<long double> == 0x1.26bb1bbb55516p+1L);
  assert(std::numbers::sqrt2_v<long double> == 0x1.6a09e667f3bcdp+0L);
  assert(std::numbers::sqrt3_v<long double> == 0x1.bb67ae8584caap+0L);
  assert(std::numbers::inv_sqrt3_v<long double> == 0x1.279a74590331cp-1L);
  assert(std::numbers::egamma_v<long double> == 0x1.2788cfc6fb619p-1L);
  assert(std::numbers::phi_v<long double> == 0x1.9e3779b97f4a8p+0L);
#elif defined(TEST_LONG_DOUBLE_IS_80_BIT)
  assert(std::numbers::e_v<long double> == 0x1.5bf0a8b145769536p+1L);
  assert(std::numbers::log2e_v<long double> == 0x1.71547652b82fe178p+0L);
  assert(std::numbers::log10e_v<long double> == 0x1.bcb7b1526e50e32ap-2L);
  assert(std::numbers::pi_v<long double> == 0x1.921fb54442d1846ap+1L);
  assert(std::numbers::inv_pi_v<long double> == 0x1.45f306dc9c882a54p-2L);
  assert(std::numbers::inv_sqrtpi_v<long double> == 0x1.20dd750429b6d11ap-1L);
  assert(std::numbers::ln2_v<long double> == 0x1.62e42fefa39ef358p-1L);
  assert(std::numbers::ln10_v<long double> == 0x1.26bb1bbb5551582ep+1L);
  assert(std::numbers::sqrt2_v<long double> == 0x1.6a09e667f3bcc908p+0L);
  assert(std::numbers::sqrt3_v<long double> == 0x1.bb67ae8584caa73cp+0L);
  assert(std::numbers::inv_sqrt3_v<long double> == 0x1.279a74590331c4d2p-1L);
  assert(std::numbers::egamma_v<long double> == 0x1.2788cfc6fb618f4ap-1L);
  assert(std::numbers::phi_v<long double> == 0x1.9e3779b97f4a7c16p+0L);
#elif defined(TEST_LONG_DOUBLE_IS_BINARY128)
  assert(std::numbers::e_v<long double> == 0x1.5bf0a8b1457695355fb8ac404e7ap+1L);
  assert(std::numbers::log2e_v<long double> == 0x1.71547652b82fe1777d0ffda0d23ap+0L);
  assert(std::numbers::log10e_v<long double> == 0x1.bcb7b1526e50e32a6ab7555f5a68p-2L);
  assert(std::numbers::pi_v<long double> == 0x1.921fb54442d18469898cc51701b8p+1L);
  assert(std::numbers::inv_pi_v<long double> == 0x1.45f306dc9c882a53f84eafa3ea6ap-2L);
  assert(std::numbers::inv_sqrtpi_v<long double> == 0x1.20dd750429b6d11ae3a914fed7fep-1L);
  assert(std::numbers::ln2_v<long double> == 0x1.62e42fefa39ef35793c7673007e6p-1L);
  assert(std::numbers::ln10_v<long double> == 0x1.26bb1bbb5551582dd4adac5705a6p+1L);
  assert(std::numbers::sqrt2_v<long double> == 0x1.6a09e667f3bcc908b2fb1366ea95p+0L);
  assert(std::numbers::sqrt3_v<long double> == 0x1.bb67ae8584caa73b25742d7078b8p+0L);
  assert(std::numbers::inv_sqrt3_v<long double> == 0x1.279a74590331c4d218f81e4afb25p-1L);
  assert(std::numbers::egamma_v<long double> == 0x1.2788cfc6fb618f49a37c7f0202a6p-1L);
  assert(std::numbers::phi_v<long double> == 0x1.9e3779b97f4a7c15f39cc0605ceep+0L);
#elif defined(TEST_LONG_DOUBLE_IS_PPCDOUBLEDOUBLE)
  // TODO: These values may be off by a few ULP since APFloat only uses 106 bits
  // of precision for PPCDoubleDouble while PPCDoubleDouble requires 2098 bits
  // (1023 - -1074 + 1) bits of precision for round trip conversion.
  //
  // TODO: Ideally we would extract the high and low halves for the
  // PPCDoubleDouble and test those individually.
  assert(std::numbers::e_v<long double> == 0x1.5bf0a8b1457695355fb8ac404e7ap+1L);
  assert(std::numbers::log2e_v<long double> == 0x1.71547652b82fe1777d0ffda0d23ap+0L);
  assert(std::numbers::log10e_v<long double> == 0x1.bcb7b1526e50e32a6ab7555f5a68p-2L);
  assert(std::numbers::pi_v<long double> == 0x1.921fb54442d18469898cc51701b8p+1L);
  assert(std::numbers::inv_pi_v<long double> == 0x1.45f306dc9c882a53f84eafa3ea6ap-2L);
  assert(std::numbers::inv_sqrtpi_v<long double> == 0x1.20dd750429b6d11ae3a914fed7fep-1L);
  assert(std::numbers::ln2_v<long double> == 0x1.62e42fefa39ef35793c7673007e6p-1L);
  assert(std::numbers::ln10_v<long double> == 0x1.26bb1bbb5551582dd4adac5705a6p+1L);
  assert(std::numbers::sqrt2_v<long double> == 0x1.6a09e667f3bcc908b2fb1366ea95p+0L);
  assert(std::numbers::sqrt3_v<long double> == 0x1.bb67ae8584caa73b25742d7078b8p+0L);
  assert(std::numbers::inv_sqrt3_v<long double> == 0x1.279a74590331c4d218f81e4afb25p-1L);
  assert(std::numbers::egamma_v<long double> == 0x1.2788cfc6fb618f49a37c7f0202a6p-1L);
  assert(std::numbers::phi_v<long double> == 0x1.9e3779b97f4a7c15f39cc0605ceep+0L);
#else
#  error "Unknown long double format"
#endif

  return true;
}

static_assert(tests());

int main(int, char**) {
  tests();
  return 0;
}
