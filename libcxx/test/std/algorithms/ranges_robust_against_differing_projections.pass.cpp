//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>
//
// Range algorithms should support the case where the ranges they operate on have different value types and the given
// projection functors are different (each projection applies to a different value type).

#include <algorithm>

#include <array>
#include <functional>
#include <iterator>
#include <ranges>
#include <utility>

// (in1, in2, ...)
template <class Func, std::ranges::range Input1, std::ranges::range Input2, class ...Args>
constexpr void test(Func&& func, Input1& in1, Input2& in2, Args&& ...args) {
  (void)func(in1.begin(), in1.end(), in2.begin(), in2.end(), std::forward<Args>(args)...);
  (void)func(in1, in2, std::forward<Args>(args)...);
}

constexpr bool test_all() {
  struct A {
    int x = 0;

    constexpr A() = default;
    constexpr A(int value) : x(value) {}
    constexpr operator int() const { return x; }

    constexpr auto operator<=>(const A&) const = default;
  };

  std::array in = {1, 2, 3};
  std::array in2 = {A{4}, A{5}, A{6}};

  std::array output = {7, 8, 9, 10, 11, 12};
  auto out = output.begin();
  std::array output2 = {A{7}, A{8}, A{9}, A{10}, A{11}, A{12}};
  auto out2 = output2.begin();

  std::ranges::equal_to eq;
  std::ranges::less less;
  auto sum = [](int lhs, A rhs) { return lhs + rhs.x; };
  auto proj1 = [](int x) { return x * -1; };
  auto proj2 = [](A a) { return a.x * -1; };

  test(std::ranges::equal, in, in2, eq, proj1, proj2);
  test(std::ranges::lexicographical_compare, in, in2, eq, proj1, proj2);
  test(std::ranges::is_permutation, in, in2, eq, proj1, proj2);
  test(std::ranges::includes, in, in2, less, proj1, proj2);
  test(std::ranges::find_first_of, in, in2, eq, proj1, proj2);
  test(std::ranges::mismatch, in, in2, eq, proj1, proj2);
  test(std::ranges::search, in, in2, eq, proj1, proj2);
  test(std::ranges::find_end, in, in2, eq, proj1, proj2);
  test(std::ranges::transform, in, in2, out, sum, proj1, proj2);
  test(std::ranges::transform, in, in2, out2, sum, proj1, proj2);
  test(std::ranges::partial_sort_copy, in, in2, less, proj1, proj2);
  test(std::ranges::merge, in, in2, out, less, proj1, proj2);
  test(std::ranges::merge, in, in2, out2, less, proj1, proj2);
  test(std::ranges::set_intersection, in, in2, out, less, proj1, proj2);
  test(std::ranges::set_intersection, in, in2, out2, less, proj1, proj2);
  test(std::ranges::set_difference, in, in2, out, less, proj1, proj2);
  test(std::ranges::set_difference, in, in2, out2, less, proj1, proj2);
  test(std::ranges::set_symmetric_difference, in, in2, out, less, proj1, proj2);
  test(std::ranges::set_symmetric_difference, in, in2, out2, less, proj1, proj2);
  test(std::ranges::set_union, in, in2, out, less, proj1, proj2);
  test(std::ranges::set_union, in, in2, out2, less, proj1, proj2);
  //test(std::ranges::starts_with, in, in2, eq, proj1, proj2);
  //test(std::ranges::ends_with, in, in2, eq, proj1, proj2);

  return true;
}

int main(int, char**) {
  test_all();
  static_assert(test_all());

  return 0;
}
