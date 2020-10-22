//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

#include <numbers>

constexpr bool tests() {
  [[maybe_unused]] const double* dd0{&std::numbers::e};
  [[maybe_unused]] const double* dd1{&std::numbers::log2e};
  [[maybe_unused]] const double* dd2{&std::numbers::log10e};
  [[maybe_unused]] const double* dd3{&std::numbers::pi};
  [[maybe_unused]] const double* dd4{&std::numbers::inv_pi};
  [[maybe_unused]] const double* dd5{&std::numbers::inv_sqrtpi};
  [[maybe_unused]] const double* dd6{&std::numbers::ln2};
  [[maybe_unused]] const double* dd7{&std::numbers::ln10};
  [[maybe_unused]] const double* dd8{&std::numbers::sqrt2};
  [[maybe_unused]] const double* dd9{&std::numbers::sqrt3};
  [[maybe_unused]] const double* dd10{&std::numbers::inv_sqrt3};
  [[maybe_unused]] const double* dd11{&std::numbers::egamma};
  [[maybe_unused]] const double* dd12{&std::numbers::phi};

  [[maybe_unused]] const float* f0{&std::numbers::e_v<float>};
  [[maybe_unused]] const float* f1{&std::numbers::log2e_v<float>};
  [[maybe_unused]] const float* f2{&std::numbers::log10e_v<float>};
  [[maybe_unused]] const float* f3{&std::numbers::pi_v<float>};
  [[maybe_unused]] const float* f4{&std::numbers::inv_pi_v<float>};
  [[maybe_unused]] const float* f5{&std::numbers::inv_sqrtpi_v<float>};
  [[maybe_unused]] const float* f6{&std::numbers::ln2_v<float>};
  [[maybe_unused]] const float* f7{&std::numbers::ln10_v<float>};
  [[maybe_unused]] const float* f8{&std::numbers::sqrt2_v<float>};
  [[maybe_unused]] const float* f9{&std::numbers::sqrt3_v<float>};
  [[maybe_unused]] const float* f10{&std::numbers::inv_sqrt3_v<float>};
  [[maybe_unused]] const float* f11{&std::numbers::egamma_v<float>};
  [[maybe_unused]] const float* f12{&std::numbers::phi_v<float>};

  [[maybe_unused]] const double* d0{&std::numbers::e_v<double>};
  [[maybe_unused]] const double* d1{&std::numbers::log2e_v<double>};
  [[maybe_unused]] const double* d2{&std::numbers::log10e_v<double>};
  [[maybe_unused]] const double* d3{&std::numbers::pi_v<double>};
  [[maybe_unused]] const double* d4{&std::numbers::inv_pi_v<double>};
  [[maybe_unused]] const double* d5{&std::numbers::inv_sqrtpi_v<double>};
  [[maybe_unused]] const double* d6{&std::numbers::ln2_v<double>};
  [[maybe_unused]] const double* d7{&std::numbers::ln10_v<double>};
  [[maybe_unused]] const double* d8{&std::numbers::sqrt2_v<double>};
  [[maybe_unused]] const double* d9{&std::numbers::sqrt3_v<double>};
  [[maybe_unused]] const double* d10{&std::numbers::inv_sqrt3_v<double>};
  [[maybe_unused]] const double* d11{&std::numbers::egamma_v<double>};
  [[maybe_unused]] const double* d12{&std::numbers::phi_v<double>};

  [[maybe_unused]] const long double* ld0{&std::numbers::e_v<long double>};
  [[maybe_unused]] const long double* ld1{&std::numbers::log2e_v<long double>};
  [[maybe_unused]] const long double* ld2{&std::numbers::log10e_v<long double>};
  [[maybe_unused]] const long double* ld3{&std::numbers::pi_v<long double>};
  [[maybe_unused]] const long double* ld4{&std::numbers::inv_pi_v<long double>};
  [[maybe_unused]] const long double* ld5{&std::numbers::inv_sqrtpi_v<long double>};
  [[maybe_unused]] const long double* ld6{&std::numbers::ln2_v<long double>};
  [[maybe_unused]] const long double* ld7{&std::numbers::ln10_v<long double>};
  [[maybe_unused]] const long double* ld8{&std::numbers::sqrt2_v<long double>};
  [[maybe_unused]] const long double* ld9{&std::numbers::sqrt3_v<long double>};
  [[maybe_unused]] const long double* ld10{&std::numbers::inv_sqrt3_v<long double>};
  [[maybe_unused]] const long double* ld11{&std::numbers::egamma_v<long double>};
  [[maybe_unused]] const long double* ld12{&std::numbers::phi_v<long double>};

  return true;
}

static_assert(tests());

int main(int, char**) {
  tests();
  return 0;
}
