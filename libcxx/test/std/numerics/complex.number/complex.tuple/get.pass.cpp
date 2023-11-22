//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <complex>

// template<size_t I, class T>
// constexpr T& get(complex<T>& __z) noexcept;

// template<size_t I, class T>
// constexpr const T& get(const complex<T>& __z) noexcept;

// template<size_t I, class T>
// constexpr T&& get(complex<T>&& __z) noexcept;

// template<size_t I, class T>
// constexpr const T&& get(const complex<T>&& __z) noexcept;

#include <complex>
#include <cassert>

template <typename T>
constexpr bool test() {
  // T&
  {
    std::complex<T> c(-1, 1);
    static_assert(std::is_same_v<decltype(std::get<0>(c)), T&>);
    static_assert(std::is_same_v<decltype(std::get<1>(c)), T&>);
    assert(std::get<0>(c) == -1);
    assert(std::get<1>(c) == 1);

    std::get<0>(c) = 2;
    assert(std::get<0>(c) == 2);

    std::get<1>(c) = -2;
    assert(std::get<0>(c) == 2);
    assert(std::get<1>(c) == -2);
  }

  // const T&
  {
    const std::complex<T> c(-1, 1);
    static_assert(std::is_same_v<decltype(std::get<0>(c)), const T&>);
    static_assert(std::is_same_v<decltype(std::get<1>(c)), const T&>);
    assert(std::get<0>(c) == -1);
    assert(std::get<1>(c) == 1);
  }

  // T&&
  {
    std::complex<T> c1(-1, 1), c2(-2, 2);
    static_assert(std::is_same_v<decltype(std::get<0>(std::move(c1))), T&&>);
    static_assert(std::is_same_v<decltype(std::get<1>(std::move(c1))), T&&>);
    assert(std::get<0>(std::move(c1)) == -1);
    assert(std::get<1>(std::move(c2)) == 2);
  }

  // const T&&
  {
    const std::complex<T> c1(-1, 1), c2(-2, 2);
    static_assert(std::is_same_v<decltype(std::get<0>(std::move(c1))), const T&&>);
    static_assert(std::is_same_v<decltype(std::get<1>(std::move(c1))), const T&&>);
    assert(std::get<0>(std::move(c1)) == -1);
    assert(std::get<1>(std::move(c2)) == 2);
  }

  return true;
}

int main() {
  test<double>();
  test<float>();
  test<long double>();
  static_assert(test<double>());
  static_assert(test<float>());
  static_assert(test<long double>());
}
