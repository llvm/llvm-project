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

// There are only a few typical signatures shared by most algorithms -- this set of helpers invokes both the
// (iterator, sentinel, ...) and the (range, ...) overloads of the given niebloid.

// (in, val)
template <class Func, std::ranges::range Input, class T, class Proj>
void in_val(Func&& func, Input& in, const T& val, Proj&& proj) {
  func(in.begin(), in.end(), val, proj);
  func(in, val, proj);
}

// (in, pred)
template <class Func, std::ranges::range Input, class Pred, class Proj>
void in_pred(Func&& func, Input& in, Pred&& pred, Proj&& proj) {
  func(in.begin(), in.end(), pred, proj);
  func(in, pred, proj);
}

// (in, val, pred)
template <class Func, std::ranges::range Input, class T, class Pred, class Proj>
void in_val_pred(Func&& func, Input& in, const T& val, Pred&& pred, Proj&& proj) {
  func(in.begin(), in.end(), val, pred, proj);
  func(in, val, pred, proj);
}

// (in1, in2, pred) -- two projections.
template <class Func, std::ranges::range Input, class Pred, class Proj1, class Proj2>
void in2_pred(Func&& func, Input& in1, Input& in2, Pred&& pred, Proj1&& proj1, Proj2&& proj2) {
  func(in1.begin(), in1.end(), in2.begin(), in2.end(), pred, proj1, proj2);
  func(in1, in2, pred, proj1, proj2);
}

// (in, out, pred)
template <class Func, std::ranges::range Input, std::weakly_incrementable Output, class Pred, class Proj>
void in_out_pred(Func&& func, Input& in, Output out, Pred&& pred, Proj&& proj) {
  func(in.begin(), in.end(), out, pred, proj);
  func(in, out, pred, proj);
}

// (in, mid, pred)
template <class Func, std::ranges::range Input, std::input_iterator Iter, class Pred, class Proj>
void in_mid_pred(Func&& func, Input& in, Iter mid, Pred&& pred, Proj&& proj) {
  func(in.begin(), mid, in.end(), pred, proj);
  func(in, mid, pred, proj);
}

// (in1, in2, out, pred) -- two projections.
template <class Func, std::ranges::range Input, std::weakly_incrementable Output, class Pred, class Proj1, class Proj2>
void in2_out_pred(Func&& func, Input& in1, Input& in2, Output out, Pred&& pred, Proj1&& proj1, Proj2&& proj2) {
  func(in1.begin(), in1.end(), in2.begin(), in2.end(), out, pred, proj1, proj2);
  func(in1, in2, out, pred, proj1, proj2);
}

void test_all() {
  std::array in = {Bar{Foo{1}}, Bar{Foo{2}}, Bar{Foo{3}}};
  std::array in2 = {Bar{Foo{4}}, Bar{Foo{5}}, Bar{Foo{6}}};
  auto mid = in.begin() + 1;

  std::array output = {Bar{Foo{4}}, Bar{Foo{5}}, Bar{Foo{6}}};
  auto out = output.begin();
  //auto out2 = output.begin() + 1;

  Bar a{Foo{1}};
  Bar b{Foo{2}};
  //Bar c{Foo{3}};

  Foo x{2};
  size_t count = 1;

  in_pred(std::ranges::any_of, in, &Foo::unary_pred, &Bar::val);
  in_pred(std::ranges::all_of, in, &Foo::unary_pred, &Bar::val);
  in_pred(std::ranges::none_of, in, &Foo::unary_pred, &Bar::val);
  in_val(std::ranges::find, in, x, &Bar::val);
  in_pred(std::ranges::find_if, in, &Foo::unary_pred, &Bar::val);
  in_pred(std::ranges::find_if_not, in, &Foo::unary_pred, &Bar::val);
  in2_pred(std::ranges::find_first_of, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  in_pred(std::ranges::adjacent_find, in, &Foo::binary_pred, &Bar::val);
  in2_pred(std::ranges::mismatch, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  in2_pred(std::ranges::equal, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  in2_pred(std::ranges::lexicographical_compare, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  //in_pred(std::ranges::partition_point, in, &Foo::unary_pred, &Bar::val);
  in_val_pred(std::ranges::lower_bound, in, x, &Foo::binary_pred, &Bar::val);
  in_val_pred(std::ranges::upper_bound, in, x, &Foo::binary_pred, &Bar::val);
  //in_val_pred(std::ranges::equal_range, in, x, &Foo::binary_pred, &Bar::val);
  in_val_pred(std::ranges::binary_search, in, x, &Foo::binary_pred, &Bar::val);

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

  in_pred(std::ranges::min_element, in, &Foo::binary_pred, &Bar::val);
  in_pred(std::ranges::max_element, in, &Foo::binary_pred, &Bar::val);
  in_pred(std::ranges::minmax_element, in, &Foo::binary_pred, &Bar::val);
  in_val(std::ranges::count, in, x, &Bar::val);
  in_pred(std::ranges::count_if, in, &Foo::unary_pred, &Bar::val);
  in2_pred(std::ranges::search, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  // search_n
  std::ranges::search_n(in.begin(), in.end(), count, x, &Foo::binary_pred, &Bar::val);
  std::ranges::search_n(in, count, x, &Foo::binary_pred, &Bar::val);
  in2_pred(std::ranges::find_end, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  in_pred(std::ranges::is_partitioned, in, &Foo::unary_pred, &Bar::val);
  in_pred(std::ranges::is_sorted, in, &Foo::binary_pred, &Bar::val);
  in_pred(std::ranges::is_sorted_until, in, &Foo::binary_pred, &Bar::val);
  //in2_pred(std::ranges::includes, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  //in_pred(std::ranges::is_heap, in, &Foo::binary_pred, &Bar::val);
  //in_pred(std::ranges::is_heap_until, in, &Foo::binary_pred, &Bar::val);
  //std::ranges::clamp(b, a, c, &Foo::binary_pred);
  //in2_pred(std::ranges::is_permutation, in, in2, &Foo::binary_pred, &Bar::val, &Bar::val);
  in_pred(std::ranges::for_each, in, &Foo::unary_pred, &Bar::val);
  std::ranges::for_each_n(in.begin(), count, &Foo::unary_pred, &Bar::val);
  // `copy`, `copy_n` and `copy_backward` have neither a projection nor a predicate.
  in_out_pred(std::ranges::copy_if, in, out, &Foo::unary_pred, &Bar::val);
  // `move` and `move_backward` have neither a projection nor a predicate.
  // `fill` and `fill_n` have neither a projection nor a predicate.
  {
    std::array out_transform = {false, true, true};
    in_out_pred(std::ranges::transform, in, out_transform.begin(), &Foo::unary_pred, &Bar::val);
  }
  // generate
  //std::ranges::generate(in.begin(), in.end(), &Bar::create);
  //std::ranges::generate(in, &Bar::create);
  // generate_n
  //std::ranges::generate(in.begin(), count, &Bar::create);
  // remove_copy
  //std::ranges::remove_copy(in.begin(), in.end(), out, x, &Bar::val);
  //std::ranges::remove_copy(in, out, x, &Bar::val);
  //in_out_pred(std::ranges::remove_copy_if, in, out, &Foo::unary_pred, &Bar::val);
  // `replace*` algorithms only use the projection to compare the elements, not to write them.
  // replace
  std::ranges::replace(in.begin(), in.end(), x, a, &Bar::val);
  std::ranges::replace(in, x, a, &Bar::val);
  // replace_if
  std::ranges::replace_if(in.begin(), in.end(), &Foo::unary_pred, a, &Bar::val);
  std::ranges::replace_if(in, &Foo::unary_pred, a, &Bar::val);
  // replace_copy
  //std::ranges::replace_copy(in.begin(), in.end(), out, x, a, &Bar::val);
  //std::ranges::replace_copy(in, out, x, a, &Bar::val);
  // replace_copy_if
  //std::ranges::replace_copy_if(in.begin(), in.end(), out, pred, a, &Bar::val);
  //std::ranges::replace_copy_if(in, out, pred, a, &Bar::val);
  // `swap_ranges` has neither a projection nor a predicate.
  // `reverse_copy` has neither a projection nor a predicate.
  // `rotate_copy` has neither a projection nor a predicate.
  // `sample` has no requirement that the given generator be invoked via `std::invoke`.
  //in_out_pred(std::ranges::unique_copy, in, out, &Foo::binary_pred, &Bar::val);
  // partition_copy
  //std::ranges::partition_copy(in.begin(), in.end(), out, out2, &Foo::unary_pred, &Bar::val);
  //std::ranges::partition_copy(in, out, out2, &Foo::unary_pred, &Bar::val);
  //in2_pred(std::ranges::partial_sort_copy, in, in2, &Foo::binary_pred, &Bar::val);
  in2_out_pred(std::ranges::merge, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  in2_out_pred(std::ranges::set_difference, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  in2_out_pred(std::ranges::set_intersection, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  in2_out_pred(std::ranges::set_symmetric_difference, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  in2_out_pred(std::ranges::set_union, in, in2, out, &Foo::binary_pred, &Bar::val, &Bar::val);
  in_val(std::ranges::remove, in, x, &Bar::val);
  in_pred(std::ranges::remove_if, in, &Foo::unary_pred, &Bar::val);
  // `reverse` has neither a projection nor a predicate.
  // `rotate` has neither a projection nor a predicate.
  // `shuffle` has neither a projection nor a predicate.
  //in_pred(std::ranges::unique, in, &Foo::binary_pred, &Bar::val);
  in_pred(std::ranges::partition, in, &Foo::unary_pred, &Bar::val);
  in_pred(std::ranges::stable_partition, in, &Foo::unary_pred, &Bar::val);
  in_pred(std::ranges::sort, in, &Foo::binary_pred, &Bar::val);
  in_pred(std::ranges::stable_sort, in, &Foo::binary_pred, &Bar::val);
  //in_mid_pred(std::ranges::partial_sort, in, mid, binary_pred);
  in_mid_pred(std::ranges::nth_element, in, mid, &Foo::binary_pred, &Bar::val);
  //in_mid_pred(std::ranges::inplace_merge, in, mid, binary_pred);
  in_pred(std::ranges::make_heap, in, &Foo::binary_pred, &Bar::val);
  in_pred(std::ranges::push_heap, in, &Foo::binary_pred, &Bar::val);
  in_pred(std::ranges::pop_heap, in, &Foo::binary_pred, &Bar::val);
  in_pred(std::ranges::sort_heap, in, &Foo::binary_pred, &Bar::val);
  //in_pred(std::ranges::prev_permutation, in, &Foo::binary_pred, &Bar::val);
  //in_pred(std::ranges::next_permutation, in, &Foo::binary_pred, &Bar::val);
}
