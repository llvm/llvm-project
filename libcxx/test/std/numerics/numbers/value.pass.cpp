//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

#include <cassert>
#include <numbers>

constexpr bool tests() {
  assert(std::numbers::e == 0x1.5bf0a8b145769p+1);
  assert(std::numbers::e_v<double> == 0x1.5bf0a8b145769p+1);
  assert(std::numbers::e_v<long double> == 0x1.5bf0a8b145769p+1l);
  assert(std::numbers::e_v<float> == 0x1.5bf0a8p+1f);

  assert(std::numbers::log2e == 0x1.71547652b82fep+0);
  assert(std::numbers::log2e_v<double> == 0x1.71547652b82fep+0);
  assert(std::numbers::log2e_v<long double> == 0x1.71547652b82fep+0l);
  assert(std::numbers::log2e_v<float> == 0x1.715476p+0f);

  assert(std::numbers::log10e == 0x1.bcb7b1526e50ep-2);
  assert(std::numbers::log10e_v<double> == 0x1.bcb7b1526e50ep-2);
  assert(std::numbers::log10e_v<long double> == 0x1.bcb7b1526e50ep-2l);
  assert(std::numbers::log10e_v<float> == 0x1.bcb7b15p-2f);

  assert(std::numbers::pi == 0x1.921fb54442d18p+1);
  assert(std::numbers::pi_v<double> == 0x1.921fb54442d18p+1);
  assert(std::numbers::pi_v<long double> == 0x1.921fb54442d18p+1l);
  assert(std::numbers::pi_v<float> == 0x1.921fb54p+1f);

  assert(std::numbers::inv_pi == 0x1.45f306dc9c883p-2);
  assert(std::numbers::inv_pi_v<double> == 0x1.45f306dc9c883p-2);
  assert(std::numbers::inv_pi_v<long double> == 0x1.45f306dc9c883p-2l);
  assert(std::numbers::inv_pi_v<float> == 0x1.45f306p-2f);

  assert(std::numbers::inv_sqrtpi == 0x1.20dd750429b6dp-1);
  assert(std::numbers::inv_sqrtpi_v<double> == 0x1.20dd750429b6dp-1);
  assert(std::numbers::inv_sqrtpi_v<long double> == 0x1.20dd750429b6dp-1l);
  assert(std::numbers::inv_sqrtpi_v<float> == 0x1.20dd76p-1f);

  assert(std::numbers::ln2 == 0x1.62e42fefa39efp-1);
  assert(std::numbers::ln2_v<double> == 0x1.62e42fefa39efp-1);
  assert(std::numbers::ln2_v<long double> == 0x1.62e42fefa39efp-1l);
  assert(std::numbers::ln2_v<float> == 0x1.62e42fp-1f);

  assert(std::numbers::ln10 == 0x1.26bb1bbb55516p+1);
  assert(std::numbers::ln10_v<double> == 0x1.26bb1bbb55516p+1);
  assert(std::numbers::ln10_v<long double> == 0x1.26bb1bbb55516p+1l);
  assert(std::numbers::ln10_v<float> == 0x1.26bb1bp+1f);

  assert(std::numbers::sqrt2 == 0x1.6a09e667f3bcdp+0);
  assert(std::numbers::sqrt2_v<double> == 0x1.6a09e667f3bcdp+0);
  assert(std::numbers::sqrt2_v<long double> == 0x1.6a09e667f3bcdp+0l);
  assert(std::numbers::sqrt2_v<float> == 0x1.6a09e6p+0f);

  assert(std::numbers::sqrt3 == 0x1.bb67ae8584caap+0);
  assert(std::numbers::sqrt3_v<double> == 0x1.bb67ae8584caap+0);
  assert(std::numbers::sqrt3_v<long double> == 0x1.bb67ae8584caap+0l);
  assert(std::numbers::sqrt3_v<float> == 0x1.bb67aep+0f);

  assert(std::numbers::inv_sqrt3 == 0x1.279a74590331cp-1);
  assert(std::numbers::inv_sqrt3_v<double> == 0x1.279a74590331cp-1);
  assert(std::numbers::inv_sqrt3_v<long double> == 0x1.279a74590331cp-1l);
  assert(std::numbers::inv_sqrt3_v<float> == 0x1.279a74p-1f);

  assert(std::numbers::egamma == 0x1.2788cfc6fb619p-1);
  assert(std::numbers::egamma_v<double> == 0x1.2788cfc6fb619p-1);
  assert(std::numbers::egamma_v<long double> == 0x1.2788cfc6fb619p-1l);
  assert(std::numbers::egamma_v<float> == 0x1.2788cfp-1f);

  assert(std::numbers::phi == 0x1.9e3779b97f4a8p+0);
  assert(std::numbers::phi_v<double> == 0x1.9e3779b97f4a8p+0);
  assert(std::numbers::phi_v<long double> == 0x1.9e3779b97f4a8p+0l);
  assert(std::numbers::phi_v<float> == 0x1.9e3779ap+0f);

  return true;
}

static_assert(tests());

int main() { tests(); }
