//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <complex>

//   template<size_t I, class T>
//     constexpr T& get(complex<T>&) noexcept;
//   template<size_t I, class T>
//     constexpr T&& get(complex<T>&&) noexcept;
//   template<size_t I, class T>
//     constexpr const T& get(const complex<T>&) noexcept;
//   template<size_t I, class T>
//     constexpr const T&& get(const complex<T>&&) noexcept;

#include <cassert>
#include <complex>
#include <utility>

template <typename T>
void test() {
  using C = std::complex<T>;

  // &
  {
    C c{T{27}, T{28}};
    // expected-error-re@*:* 3{{static assertion failed {{.*}}Index value is out of range.}}
    std::get<3>(c);
  }
  // &&
  {
    C c{T{27}, T{28}};
    // expected-error-re@*:* 3 {{static assertion failed {{.*}}Index value is out of range.}}
    std::get<3>(std::move(c));
  }
  // const &
  {
    const C c{T{27}, T{28}};
    // expected-error-re@*:* 3 {{static assertion failed {{.*}}Index value is out of range.}}
    std::get<3>(c);
  }
  // const &&
  {
    const C c{T{27}, T{28}};
    // expected-error-re@*:* 3 {{static assertion failed {{.*}}Index value is out of range.}}
    std::get<3>(std::move(c));
  }
}

void test() {
  test<float>();
  test<double>();
  test<long double>();
}
