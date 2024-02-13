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
#include <concepts>
#include <complex>
#include <utility>

template <typename T>
constexpr void test() {
  // Structured binding

  // &
  {
    std::complex<T> c{T{27}, T{28}};

    auto& [r, i]{c};
    assert(r == T{27});
    assert(i == T{28});
  }
  //  &&
  {
    std::complex<T> c{T{27}, T{28}};

    auto&& [r, i]{std::move(c)};
    assert(r == T{27});
    assert(i == T{28});
  }
  // const &
  {
    const std::complex<T> c{T{27}, T{28}};

    const auto& [r, i]{c};
    assert(r == T{27});
    assert(i == T{28});
  }
  //  const &&
  {
    const std::complex<T> c{T{27}, T{28}};

    const auto&& [r, i]{std::move(c)};
    assert(r == T{27});
    assert(i == T{28});
  }

  // Ranges
}

constexpr bool test() {
  test<float>();
  test<double>();
  test<long double>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
