//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<container-compatible-range<T> R>
//   constexpr inplace_vector(from_range_t, R&& rg);

#include <array>
#include <inplace_vector>
#include <ranges>

void fn() {
  std::array<int, 6> arr{1, 2, 3, 4, 5, 6};
  std::array<int, 1> arr2{1};

  std::inplace_vector<int, 5> v1(
      std::from_range,
      arr); // expected-error-re@inplace_vector:* {{static assertion failed{{.*}}inplace_vector<Tp,N>(from_range_t, Range): Statically sized range must be <= Capacity}}
  std::inplace_vector<int, 0> v2(
      std::from_range,
      arr2); // expected-error-re@inplace_vector:* {{static assertion failed{{.*}}inplace_vector<Tp,0>(from_range_t, Range): Statically sized range must be 0}}
}
