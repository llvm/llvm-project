//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <algorithm>
//
// Range algorithms should use `std::invoke` to call the given projection(s) (and predicates, where applicable).

#include <algorithm>

#include <array>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <ranges>

struct Foo {
  int val;
  constexpr bool unary_pred() const { return val > 0; }
  constexpr bool binary_pred(const Foo& rhs) const { return val < rhs.val; }
  constexpr auto operator<=>(const Foo&) const = default;
};

struct Bar {
  Foo val;
  Bar create() const { return Bar(); }
};

// Invokes both the (iterator, sentinel, ...) and the (range, ...) overloads of the given niebloid.

// (in, ...)
template <class Func, std::ranges::range Input, class ...Args>
constexpr void test(Func&& func, Input& in, Args&& ...args) {
  func(in.begin(), in.end(), std::forward<Args>(args)...);
  func(in, std::forward<Args>(args)...);
}

// (in1, in2, ...)
template <class Func, std::ranges::range Input, class ...Args>
constexpr void test(Func&& func, Input& in1, Input& in2, Args&& ...args) {
  func(in1.begin(), in1.end(), in2.begin(), in2.end(), std::forward<Args>(args)...);
  func(in1, in2, std::forward<Args>(args)...);
}

// (in, mid, ...)
template <class Func, std::ranges::range Input, class ...Args>
constexpr void test_mid(Func&& func, Input& in, std::ranges::iterator_t<Input> mid, Args&& ...args) {
  func(in.begin(), mid, in.end(), std::forward<Args>(args)...);
  func(in, mid, std::forward<Args>(args)...);
}

constexpr bool test_all() {
  std::array in = {Bar{Foo{1}}, Bar{Foo{2}}, Bar{Foo{3}}};
  std::array in2 = {Bar{Foo{4}}, Bar{Foo{5}}, Bar{Foo{6}}};
  auto mid = in.begin() + 1;

  std::array output = {Bar{Foo{7}}, Bar{Foo{8}}, Bar{Foo{9}}, Bar{Foo{10}}, Bar{Foo{11}}, Bar{Foo{12}}};
  auto out = output.begin();
  auto out2 = output.begin() + 1;

  Bar a{Foo{1}};
  Bar b{Foo{2}};
  //Bar c{Foo{3}};

  Foo x{2};
  size_t count = 1;

  test(std::ranges::any_of, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::all_of, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::none_of, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::find, in, x, &Bar::val);
  test(std::ranges::find_if, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::find_if_not, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::find_first_of, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::adjacent_find, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::mismatch, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::equal, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::lexicographical_compare, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::partition_point, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::lower_bound, in, x, &Foo::binary_pred, &Bar::val);
  test(std::ranges::upper_bound, in, x, &Foo::binary_pred, &Bar::val);
  test(std::ranges::equal_range, in, x, &Foo::binary_pred, &Bar::val);
  test(std::ranges::binary_search, in, x, &Foo::binary_pred, &Bar::val);

  // min
  std::ranges::min(a, b, &Foo::binary_pred, &Bar::val);
  std::ranges::min(std::initializer_list<Bar>{a, b}, &Foo::binary_pred, &Bar::val);
  std::ranges::min(in, &Foo::binary_pred, &Bar::val);
  // max
  std::ranges::max(a, b, &Foo::binary_pred, &Bar::val);
  std::ranges::max(std::initializer_list<Bar>{a, b}, &Foo::binary_pred, &Bar::val);
  std::ranges::max(in, &Foo::binary_pred, &Bar::val);
  // minmax
  std::ranges::minmax(a, b, &Foo::binary_pred, &Bar::val);
  std::ranges::minmax(std::initializer_list<Bar>{a, b}, &Foo::binary_pred, &Bar::val);
  std::ranges::minmax(in, &Foo::binary_pred, &Bar::val);

  test(std::ranges::min_element, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::max_element, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::minmax_element, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::count, in, x, &Bar::val);
  test(std::ranges::count_if, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::search, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::search_n, in, count, x, &Foo::binary_pred, &Bar::val);
  test(std::ranges::find_end, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::is_partitioned, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::is_sorted, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::is_sorted_until, in, &Foo::binary_pred, &Bar::val);
  //test(std::ranges::includes, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  //test(std::ranges::is_heap, in, &Foo::binary_pred, &Bar::val);
  //test(std::ranges::is_heap_until, in, &Foo::binary_pred, &Bar::val);
  //std::ranges::clamp(b, a, c, &Foo::binary_pred);
  //test(std::ranges::is_permutation, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::for_each, in, &Foo::unary_pred, &Bar::val);
  std::ranges::for_each_n(in.begin(), count, &Foo::unary_pred, &Bar::val);
  // `copy`, `copy_n` and `copy_backward` have neither a projection nor a predicate.
  test(std::ranges::copy_if, in, out, &Foo::unary_pred, &Bar::val);
  // `move` and `move_backward` have neither a projection nor a predicate.
  // `fill` and `fill_n` have neither a projection nor a predicate.
  {
    std::array out_transform = {false, true, true};
    test(std::ranges::transform, in, out_transform.begin(), &Foo::unary_pred, &Bar::val);
  }
  //test(std::ranges::generate, in, &Bar::create);
  //std::ranges::generate_n(in.begin(), count, &Bar::create);
  //test(std::ranges::remove_copy, in, out, x, &Bar::val);
  //test(std::ranges::remove_copy_if, in, out, &Foo::unary_pred, &Bar::val);
  // `replace*` algorithms only use the projection to compare the elements, not to write them.
  test(std::ranges::replace, in, x, a, &Bar::val);
  test(std::ranges::replace_if, in, &Foo::unary_pred, a, &Bar::val);
  //test(std::ranges::replace_copy, in, out, x, a, &Bar::val);
  //test(std::ranges::replace_copy_if, in, out, pred, a, &Bar::val);
  // `swap_ranges` has neither a projection nor a predicate.
  // `reverse_copy` has neither a projection nor a predicate.
  // `rotate_copy` has neither a projection nor a predicate.
  // `sample` has no requirement that the given generator be invoked via `std::invoke`.
  //test(std::ranges::unique_copy, in, out, &Foo::binary_pred, &Bar::val);
  test(std::ranges::partition_copy, in, out, out2, &Foo::unary_pred, &Bar::val);
  //test(std::ranges::partial_sort_copy, in, in2, &Foo::binary_pred, &Bar::val);
  test(std::ranges::merge, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::set_difference, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::set_intersection, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::set_symmetric_difference, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::set_union, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  test(std::ranges::remove, in, x, &Bar::val);
  test(std::ranges::remove_if, in, &Foo::unary_pred, &Bar::val);
  // `reverse` has neither a projection nor a predicate.
  // `rotate` has neither a projection nor a predicate.
  // `shuffle` has neither a projection nor a predicate.
  //test(std::ranges::unique, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::partition, in, &Foo::unary_pred, &Bar::val);
  if (!std::is_constant_evaluated())
    test(std::ranges::stable_partition, in, &Foo::unary_pred, &Bar::val);
  test(std::ranges::sort, in, &Foo::binary_pred, &Bar::val);
  if (!std::is_constant_evaluated())
    test(std::ranges::stable_sort, in, &Foo::binary_pred, &Bar::val);
  test_mid(std::ranges::partial_sort, in, mid, &Foo::binary_pred, &Bar::val);
  test_mid(std::ranges::nth_element, in, mid, &Foo::binary_pred, &Bar::val);
  //test_mid(std::ranges::inplace_merge, in, mid, binary_pred);
  test(std::ranges::make_heap, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::push_heap, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::pop_heap, in, &Foo::binary_pred, &Bar::val);
  test(std::ranges::sort_heap, in, &Foo::binary_pred, &Bar::val);
  //test(std::ranges::prev_permutation, in, &Foo::binary_pred, &Bar::val);
  //test(std::ranges::next_permutation, in, &Foo::binary_pred, &Bar::val);

  return true;
}

int main(int, char**) {
  test_all();
  static_assert(test_all());

  return 0;
}
