//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

#include <cassert>
#include <numbers>

// We are testing if template instantiation works. Don't care if the result is unused.
#if defined(__clang__)
#   pragma clang diagnostic ignored "-Wunused-variable"
#endif

constexpr bool tests() {
  float f0{std::numbers::e_v<float>};
  float f1{std::numbers::log2e_v<float>};
  float f2{std::numbers::log10e_v<float>};
  float f3{std::numbers::pi_v<float>};
  float f4{std::numbers::inv_pi_v<float>};
  float f5{std::numbers::inv_sqrtpi_v<float>};
  float f6{std::numbers::ln2_v<float>};
  float f7{std::numbers::ln10_v<float>};
  float f8{std::numbers::sqrt2_v<float>};
  float f9{std::numbers::sqrt3_v<float>};
  float f10{std::numbers::inv_sqrt3_v<float>};
  float f11{std::numbers::egamma_v<float>};
  float f12{std::numbers::phi_v<float>};

  double d0{std::numbers::e_v<double>};
  double d1{std::numbers::log2e_v<double>};
  double d2{std::numbers::log10e_v<double>};
  double d3{std::numbers::pi_v<double>};
  double d4{std::numbers::inv_pi_v<double>};
  double d5{std::numbers::inv_sqrtpi_v<double>};
  double d6{std::numbers::ln2_v<double>};
  double d7{std::numbers::ln10_v<double>};
  double d8{std::numbers::sqrt2_v<double>};
  double d9{std::numbers::sqrt3_v<double>};
  double d10{std::numbers::inv_sqrt3_v<double>};
  double d11{std::numbers::egamma_v<double>};
  double d12{std::numbers::phi_v<double>};

  assert(d0 == std::numbers::e);
  assert(d1 == std::numbers::log2e);
  assert(d2 == std::numbers::log10e);
  assert(d3 == std::numbers::pi);
  assert(d4 == std::numbers::inv_pi);
  assert(d5 == std::numbers::inv_sqrtpi);
  assert(d6 == std::numbers::ln2);
  assert(d7 == std::numbers::ln10);
  assert(d8 == std::numbers::sqrt2);
  assert(d9 == std::numbers::sqrt3);
  assert(d10 == std::numbers::inv_sqrt3);
  assert(d11 == std::numbers::egamma);
  assert(d12 == std::numbers::phi);

  long double ld0{std::numbers::e_v<long double>};
  long double ld1{std::numbers::log2e_v<long double>};
  long double ld2{std::numbers::log10e_v<long double>};
  long double ld3{std::numbers::pi_v<long double>};
  long double ld4{std::numbers::inv_pi_v<long double>};
  long double ld5{std::numbers::inv_sqrtpi_v<long double>};
  long double ld6{std::numbers::ln2_v<long double>};
  long double ld7{std::numbers::ln10_v<long double>};
  long double ld8{std::numbers::sqrt2_v<long double>};
  long double ld9{std::numbers::sqrt3_v<long double>};
  long double ld10{std::numbers::inv_sqrt3_v<long double>};
  long double ld11{std::numbers::egamma_v<long double>};
  long double ld12{std::numbers::phi_v<long double>};

  return true;
}

static_assert(tests());

int main() { tests(); }
