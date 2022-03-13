//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <numbers>

struct user {
  int value;
};

template <>
user std::numbers::e_v<user>{};

template <>
user std::numbers::log2e_v<user>{};

template <>
user std::numbers::log10e_v<user>{};

template <>
user std::numbers::pi_v<user>{};

template <>
user std::numbers::inv_pi_v<user>{};

template <>
user std::numbers::inv_sqrtpi_v<user>{};

template <>
user std::numbers::ln2_v<user>{};

template <>
user std::numbers::ln10_v<user>{};

template <>
user std::numbers::sqrt2_v<user>{};

template <>
user std::numbers::sqrt3_v<user>{};

template <>
user std::numbers::inv_sqrt3_v<user>{};

template <>
user std::numbers::egamma_v<user>{};

template <>
user std::numbers::phi_v<user>{};

int main(int, char**) { return 0; }
