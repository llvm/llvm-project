//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr decltype(auto) operator*() const;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>
#include <utility>

template <std::size_t N, class T, std::size_t Size>
constexpr void testReference(T (&ts)[Size]) {
  auto ev = ts | std::views::elements<N>;
  auto it = ev.begin();

  decltype(auto) result = *it;
  using ExpectedType    = decltype(std::get<N>(ts[0]));
  static_assert(std::is_same_v<decltype(result), ExpectedType>);

  if constexpr (std::is_reference_v<ExpectedType>) {
    // tuple/array/pair
    assert(&result == &std::get<N>(ts[0]));
  } else {
    // subrange
    assert(result == std::get<N>(ts[0]));
  }
}

// LWG 3502 elements_view should not be allowed to return dangling references
template <std::size_t N, class T>
constexpr void testValue(T t) {
  auto ev = std::views::iota(0, 1) | std::views::transform([&t](int) { return t; }) | std::views::elements<N>;
  auto it = ev.begin();

  decltype(auto) result = *it;
  using ExpectedType    = std::remove_cvref_t<decltype(std::get<N>(t))>;
  static_assert(std::is_same_v<decltype(result), ExpectedType>);

  assert(result == std::get<N>(t));
}

constexpr bool test() {
  // test tuple
  {
    std::tuple<int, short, long> ts[] = {{1, short{2}, 3}, {4, short{5}, 6}};
    testReference<0>(ts);
    testReference<1>(ts);
    testReference<2>(ts);
    testValue<0>(ts[0]);
    testValue<1>(ts[0]);
    testValue<2>(ts[0]);
  }

  // test pair
  {
    std::pair<int, short> ps[] = {{1, short{2}}, {4, short{5}}};
    testReference<0>(ps);
    testReference<1>(ps);
    testValue<0>(ps[0]);
    testValue<1>(ps[0]);
  }

  // test array
  {
    std::array<int, 3> arrs[] = {{1, 2, 3}, {3, 4, 5}};
    testReference<0>(arrs);
    testReference<1>(arrs);
    testReference<2>(arrs);
    testValue<0>(arrs[0]);
    testValue<1>(arrs[0]);
    testValue<2>(arrs[0]);
  }

  // test subrange
  {
    int i                             = 5;
    std::ranges::subrange<int*> srs[] = {{&i, &i}, {&i, &i}};
    testReference<0>(srs);
    testReference<1>(srs);
    testValue<0>(srs[0]);
    testValue<1>(srs[0]);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
