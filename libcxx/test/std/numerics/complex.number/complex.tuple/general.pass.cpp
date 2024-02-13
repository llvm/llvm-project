//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <complex>

// Support for:
//  - Structured bindings
//  - Ranges

#include <cassert>
#include <complex>
#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

template <typename T>
constexpr void test() {
  // Structured binding

  // &
  {
    std::complex<T> c{T{27}, T{28}};

    auto& [r, i]{c};
    static_assert(std::same_as<T&, decltype(r)>);
    assert(r == T{27});
    static_assert(std::same_as<T&, decltype(i)>);
    assert(i == T{28});
  }
  //  &&
  {
    std::complex<T> c{T{27}, T{28}};

    auto&& [r, i]{std::move(c)};
    static_assert(std::same_as<T&&, decltype(r)>);
    assert(r == T{27});
    static_assert(std::same_as<T&&, decltype(i)>);
    assert(i == T{28});
  }
  // const &
  {
    const std::complex<T> c{T{27}, T{28}};

    const auto& [r, i]{c};
    static_assert(std::same_as<const T&, decltype(r)>);
    assert(r == T{27});
    static_assert(std::same_as<const T&, decltype(i)>);
    assert(i == T{28});
  }
  //  const &&
  {
    const std::complex<T> c{T{27}, T{28}};

    const auto&& [r, i]{std::move(c)};
    static_assert(std::same_as<const T&&, decltype(r)>);
    assert(r == T{27});
    static_assert(std::same_as<const T&&, decltype(i)>);
    assert(i == T{28});
  }

  // Ranges

  {
    std::complex<T> arr[]{{T{27}, T{28}}, {T{82}, T{94}}};

    std::same_as<std::vector<T>> decltype(auto) reals{arr | std::views::elements<0> | std::ranges::to<std::vector>()};
    assert(reals.size() == 2);
    assert(reals[0] == T{27});
    assert(reals[1] == T{82});

    std::same_as<std::vector<T>> decltype(auto) imags{arr | std::views::elements<1> | std::ranges::to<std::vector>()};
    assert(reals.size() == 2);
    assert(imags[0] == T{28});
    assert(imags[1] == T{94});
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
