//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

#include <numbers>

// We are testing if these are defined by taking an address. Don't care if the result is unused.
#if defined(__clang__)
#   pragma clang diagnostic ignored "-Wunused-variable"
#endif

constexpr bool tests() {
  const double* dd0{&std::numbers::e};
  const double* dd1{&std::numbers::log2e};
  const double* dd2{&std::numbers::log10e};
  const double* dd3{&std::numbers::pi};
  const double* dd4{&std::numbers::inv_pi};
  const double* dd5{&std::numbers::inv_sqrtpi};
  const double* dd6{&std::numbers::ln2};
  const double* dd7{&std::numbers::ln10};
  const double* dd8{&std::numbers::sqrt2};
  const double* dd9{&std::numbers::sqrt3};
  const double* dd10{&std::numbers::inv_sqrt3};
  const double* dd11{&std::numbers::egamma};
  const double* dd12{&std::numbers::phi};

  const float* f0{&std::numbers::e_v<float>};
  const float* f1{&std::numbers::log2e_v<float>};
  const float* f2{&std::numbers::log10e_v<float>};
  const float* f3{&std::numbers::pi_v<float>};
  const float* f4{&std::numbers::inv_pi_v<float>};
  const float* f5{&std::numbers::inv_sqrtpi_v<float>};
  const float* f6{&std::numbers::ln2_v<float>};
  const float* f7{&std::numbers::ln10_v<float>};
  const float* f8{&std::numbers::sqrt2_v<float>};
  const float* f9{&std::numbers::sqrt3_v<float>};
  const float* f10{&std::numbers::inv_sqrt3_v<float>};
  const float* f11{&std::numbers::egamma_v<float>};
  const float* f12{&std::numbers::phi_v<float>};

  const double* d0{&std::numbers::e_v<double>};
  const double* d1{&std::numbers::log2e_v<double>};
  const double* d2{&std::numbers::log10e_v<double>};
  const double* d3{&std::numbers::pi_v<double>};
  const double* d4{&std::numbers::inv_pi_v<double>};
  const double* d5{&std::numbers::inv_sqrtpi_v<double>};
  const double* d6{&std::numbers::ln2_v<double>};
  const double* d7{&std::numbers::ln10_v<double>};
  const double* d8{&std::numbers::sqrt2_v<double>};
  const double* d9{&std::numbers::sqrt3_v<double>};
  const double* d10{&std::numbers::inv_sqrt3_v<double>};
  const double* d11{&std::numbers::egamma_v<double>};
  const double* d12{&std::numbers::phi_v<double>};

  const long double* ld0{&std::numbers::e_v<long double>};
  const long double* ld1{&std::numbers::log2e_v<long double>};
  const long double* ld2{&std::numbers::log10e_v<long double>};
  const long double* ld3{&std::numbers::pi_v<long double>};
  const long double* ld4{&std::numbers::inv_pi_v<long double>};
  const long double* ld5{&std::numbers::inv_sqrtpi_v<long double>};
  const long double* ld6{&std::numbers::ln2_v<long double>};
  const long double* ld7{&std::numbers::ln10_v<long double>};
  const long double* ld8{&std::numbers::sqrt2_v<long double>};
  const long double* ld9{&std::numbers::sqrt3_v<long double>};
  const long double* ld10{&std::numbers::inv_sqrt3_v<long double>};
  const long double* ld11{&std::numbers::egamma_v<long double>};
  const long double* ld12{&std::numbers::phi_v<long double>};

  return true;
}

static_assert(tests());

int main() { tests(); }
