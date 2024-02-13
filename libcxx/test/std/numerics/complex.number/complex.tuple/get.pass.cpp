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
#include <concepts>
#include <utility>

template <typename T>
constexpr void test() {
  // &
  {
    std::complex<T> c{T{27}, T{28}};

    std::same_as<T&> decltype(auto) r = get<0>(c);
    static_assert(noexcept(get<0>(c)));
    assert(r == T{27});
    std::same_as<T&> decltype(auto) i = get<1>(c);
    static_assert(noexcept(get<1>(c)));
    assert(i == T{28});
  }
  //  &&
  {
    std::complex<T> c{T{27}, T{28}};

    std::same_as<T&&> decltype(auto) r = get<0>(std::move(c));
    static_assert(noexcept(get<0>(c)));
    assert(r == T{27});
  }
  {
    std::complex<T> c{T{27}, T{28}};

    std::same_as<T&&> decltype(auto) i = get<1>(std::move(c));
    static_assert(noexcept(get<1>(c)));
    assert(i == T{28});
  }
  // const &
  {
    const std::complex<T> c{T{27}, T{28}};

    std::same_as<const T&> decltype(auto) r = get<0>(c);
    static_assert(noexcept(get<0>(c)));
    assert(r == T{27});
    std::same_as<const T&> decltype(auto) i = get<1>(c);
    static_assert(noexcept(get<1>(c)));
    assert(i == T{28});
  }
  //  const &&
  {
    const std::complex<T> c{T{27}, T{28}};

    std::same_as<const T&&> decltype(auto) r = get<0>(std::move(c));
    static_assert(noexcept(get<0>(c)));
    assert(r == T{27});
  }
  {
    const std::complex<T> c{T{27}, T{28}};

    std::same_as<const T&&> decltype(auto) i = get<1>(std::move(c));
    static_assert(noexcept(get<1>(c)));
    assert(i == T{28});
  }
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
