//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Checks that `std::ranges::fold_left_with_iter`'s requirements are correct.

#include <algorithm>
#include <concepts>
#include <functional>
#include <iterator>
#include <ranges>

#include "test_range.h"
#include "test_iterators.h"
#include "../requirements.h"

// expected-error@*:* 19 {{no matching function for call to object of type 'const __fold_left_with_iter'}}

void test_iterator() {
  // expected-note@*:*  10 {{candidate template ignored: constraints not satisfied}}
  // expected-note@*:*  10 {{candidate function template not viable: requires 3 arguments, but 4 were provided}}

  std::ranges::fold_left_with_iter(bad_iterator_category(), std::unreachable_sentinel, 0, std::plus());
  // expected-note@*:* {{because 'bad_iterator_category' does not satisfy 'input_iterator'}}
  // expected-note@*:* {{because 'derived_from<_ITER_CONCEPT<bad_iterator_category>, input_iterator_tag>' evaluated to false}}
  // expected-note@*:* {{because 'is_base_of_v<std::input_iterator_tag, void>' evaluated to false}}

  {
    int* first;
    int* last;
    std::ranges::fold_left_with_iter(
        cpp17_input_iterator<int*>(first), cpp17_input_iterator<int*>(last), 0, std::plus());
    // expected-note@*:* {{because 'sentinel_for<cpp17_input_iterator<int *>, cpp17_input_iterator<int *> >' evaluated to false}}
    // expected-note@*:* {{because 'cpp17_input_iterator<int *>' does not satisfy 'semiregular'}}
    // expected-note@*:* {{'cpp17_input_iterator<int *>' does not satisfy 'default_initializable'}}
    // expected-note@*:* {{because 'cpp17_input_iterator<int *>' does not satisfy 'constructible_from'}}
    // expected-note@*:* {{because 'is_constructible_v<cpp17_input_iterator<int *> >' evaluated to false}}

    std::ranges::fold_left_with_iter(first, last, 0, non_copy_constructible_callable());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<non_copy_constructible_callable, int, int *>' evaluated to false}}
    // expected-note@*:* {{because 'non_copy_constructible_callable' does not satisfy 'copy_constructible'}}
    // expected-note@*:* {{because 'constructible_from<non_copy_constructible_callable, non_copy_constructible_callable &>' evaluated to false}}
    // expected-note@*:* {{because 'is_constructible_v<non_copy_constructible_callable, non_copy_constructible_callable &>' evaluated to false}}

    std::ranges::fold_left_with_iter(first, last, 0, not_invocable());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<not_invocable, int, int *>' evaluated to false}}
    // expected-note@*:* {{because 'invocable<not_invocable &, int, iter_reference_t<int *> >' evaluated to false}}
    // expected-note@*:* {{because 'std::invoke(std::forward<_Fn>(__fn), std::forward<_Args>(__args)...)' would be invalid: no matching function for call to 'invoke'}}
  }
  {
    S* first = nullptr;
    S* last  = nullptr;
    std::ranges::fold_left_with_iter(first, last, S(), non_decayable_result());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<non_decayable_result, S, S *>' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<non_decayable_result, S, S *, invoke_result_t<non_decayable_result &, S, iter_reference_t<S *> > >' evaluated to false}}
    // expected-note@*:* {{because 'convertible_to<volatile S &, S>' evaluated to false}}
    // expected-note@*:* {{because 'is_convertible_v<volatile S &, S>' evaluated to false}}
  }
  {
    copyable_non_movable* first;
    copyable_non_movable* last;
    std::ranges::fold_left_with_iter(first, last, non_movable(), std::plus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::plus<void>, non_movable, copyable_non_movable *>' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::plus<void>, non_movable, copyable_non_movable *, invoke_result_t<plus<void> &, non_movable, iter_reference_t<copyable_non_movable *> > >' evaluated to false}}
    // expected-note@*:* {{because 'non_movable' does not satisfy 'movable'}}
    // expected-note@*:* {{because 'non_movable' does not satisfy 'move_constructible'}}
    // expected-note@*:* {{because 'constructible_from<non_movable, non_movable>' evaluated to false}}
    // expected-note@*:* {{because 'is_constructible_v<non_movable, non_movable>' evaluated to false}}
  }
  {
    copyable_non_movable* first;
    copyable_non_movable* last;
    std::ranges::fold_left_with_iter(first, last, 0, std::minus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::minus<void>, int, copyable_non_movable *>' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::minus<void>, int, copyable_non_movable *, invoke_result_t<minus<void> &, int, iter_reference_t<copyable_non_movable *> > >' evaluated to false}}
    // expected-note@*:* {{because 'copyable_non_movable' does not satisfy 'movable'}}
    // expected-note@*:* {{because 'copyable_non_movable' does not satisfy 'move_constructible'}}
    // expected-note@*:* {{because 'constructible_from<copyable_non_movable, copyable_non_movable>' evaluated to false}}
    // expected-note@*:* {{because 'is_constructible_v<copyable_non_movable, copyable_non_movable>' evaluated to false}}
  }
  {
    int* first = nullptr;
    int* last  = nullptr;
    std::ranges::fold_left_with_iter(first, last, not_convertible_to_int(), std::plus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::plus<void>, not_convertible_to_int, int *>' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::plus<void>, not_convertible_to_int, int *, invoke_result_t<plus<void> &, not_convertible_to_int, iter_reference_t<int *> > >' evaluated to false}}
    // expected-note@*:* {{because 'convertible_to<not_convertible_to_int, int>' evaluated to false}}
    // expected-note@*:* {{because 'is_convertible_v<not_convertible_to_int, int>' evaluated to false}}
  }
  {
    not_invocable_with_decayed* first;
    not_invocable_with_decayed* last;
    std::ranges::fold_left_with_iter(first, last, 0, std::plus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::plus<void>, int, not_invocable_with_decayed *>' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::plus<void>, int, not_invocable_with_decayed *, invoke_result_t<plus<void> &, int, iter_reference_t<not_invocable_with_decayed *> > >' evaluated to false}}
    // expected-note@*:* {{because 'invocable<std::plus<void> &, not_invocable_with_decayed, iter_reference_t<not_invocable_with_decayed *> >' evaluated to false}}
    // expected-note@*:* {{because 'std::invoke(std::forward<_Fn>(__fn), std::forward<_Args>(__args)...)' would be invalid: no matching function for call to 'invoke'}}
  }
  {
    not_assignable_to_decayed* first;
    not_assignable_to_decayed* last;
    std::ranges::fold_left_with_iter(first, last, not_assignable_to_decayed(), std::plus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::plus<void>, not_assignable_to_decayed, not_assignable_to_decayed *>' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::plus<void>, not_assignable_to_decayed, not_assignable_to_decayed *, invoke_result_t<plus<void> &, not_assignable_to_decayed, iter_reference_t<not_assignable_to_decayed *> > >' evaluated to false}}
    // expected-note@*:* {{because 'assignable_from<not_assignable_to_decayed &, invoke_result_t<plus<void> &, not_assignable_to_decayed, iter_reference_t<not_assignable_to_decayed *> > >' evaluated to false}}
    // expected-note@*:* {{because '__lhs = std::forward<_Rhs>(__rhs)' would be invalid: no viable overloaded '='}}
  }
}

void test_fold_range() {
  // expected-note@*:*  9 {{candidate template ignored: constraints not satisfied}}
  // expected-note@*:*  9 {{candidate function template not viable: requires 4 arguments, but 3 were provided}}

  {
    struct bad_range {
      bad_iterator_category begin();
      std::unreachable_sentinel_t end();
    };

    bad_range r;
    std::ranges::fold_left_with_iter(r, 0, std::plus());
    // expected-note@*:* {{because 'bad_range &' does not satisfy 'input_range'}}
    // expected-note@*:* {{because 'iterator_t<bad_range &>' (aka 'bad_iterator_category') does not satisfy 'input_iterator'}}
    // expected-note@*:* {{because 'derived_from<_ITER_CONCEPT<bad_iterator_category>, input_iterator_tag>' evaluated to false}}
    // expected-note@*:* {{because 'is_base_of_v<std::input_iterator_tag, void>' evaluated to false}}
  }
  {
    test_range<cpp20_input_iterator, int> r;

    std::ranges::fold_left_with_iter(r, 0, non_copy_constructible_callable());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<non_copy_constructible_callable, int, iterator_t<test_range<cpp20_input_iterator, int> &> >' evaluated to false}}
    // expected-note@*:* {{because 'non_copy_constructible_callable' does not satisfy 'copy_constructible'}}
    // expected-note@*:* {{because 'constructible_from<non_copy_constructible_callable, non_copy_constructible_callable &>' evaluated to false}}
    // expected-note@*:* {{because 'is_constructible_v<non_copy_constructible_callable, non_copy_constructible_callable &>' evaluated to false}}

    std::ranges::fold_left_with_iter(r, 0, not_invocable());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<not_invocable, int, iterator_t<test_range<cpp20_input_iterator, int> &> >' evaluated to false}}
    // expected-note@*:* {{because 'invocable<not_invocable &, int, iter_reference_t<cpp20_input_iterator<int *> > >' evaluated to false}}
    // expected-note@*:* {{because 'std::invoke(std::forward<_Fn>(__fn), std::forward<_Args>(__args)...)' would be invalid: no matching function for call to 'invoke'}}
  }
  {
    test_range<cpp20_input_iterator, S> r;

    std::ranges::fold_left_with_iter(r, S(), non_decayable_result());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<non_decayable_result, S, iterator_t<test_range<cpp20_input_iterator, S> &> >' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<non_decayable_result, S, cpp20_input_iterator<S *>, invoke_result_t<non_decayable_result &, S, iter_reference_t<cpp20_input_iterator<S *> > > >' evaluated to false}}
    // expected-note@*:* {{because 'convertible_to<volatile S &, S>' evaluated to false}}
    // expected-note@*:* {{because 'is_convertible_v<volatile S &, S>' evaluated to false}}
  }
  {
    test_range<cpp20_input_iterator, copyable_non_movable> r;
    std::ranges::fold_left_with_iter(r, non_movable(), std::plus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::plus<void>, non_movable, iterator_t<test_range<cpp20_input_iterator, copyable_non_movable> &> >' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::plus<void>, non_movable, cpp20_input_iterator<copyable_non_movable *>, invoke_result_t<plus<void> &, non_movable, iter_reference_t<cpp20_input_iterator<copyable_non_movable *> > > >' evaluated to false}}
    // expected-note@*:* {{because 'non_movable' does not satisfy 'movable'}}
    // expected-note@*:* {{because 'non_movable' does not satisfy 'move_constructible'}}
    // expected-note@*:* {{because 'constructible_from<non_movable, non_movable>' evaluated to false}}
    // expected-note@*:* {{because 'is_constructible_v<non_movable, non_movable>' evaluated to false}}
  }
  {
    test_range<cpp20_input_iterator, copyable_non_movable> r;

    std::ranges::fold_left_with_iter(r, 0, std::minus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::minus<void>, int, iterator_t<test_range<cpp20_input_iterator, copyable_non_movable> &> >' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::minus<void>, int, cpp20_input_iterator<copyable_non_movable *>, invoke_result_t<minus<void> &, int, iter_reference_t<cpp20_input_iterator<copyable_non_movable *> > > >' evaluated to false}}
    // expected-note@*:* {{because 'copyable_non_movable' does not satisfy 'movable'}}
    // expected-note@*:* {{because 'copyable_non_movable' does not satisfy 'move_constructible'}}
    // expected-note@*:* {{because 'constructible_from<copyable_non_movable, copyable_non_movable>' evaluated to false}}
    // expected-note@*:* {{because 'is_constructible_v<copyable_non_movable, copyable_non_movable>' evaluated to false}}
  }
  {
    test_range<cpp20_input_iterator, int> r;
    std::ranges::fold_left_with_iter(r, not_convertible_to_int(), std::plus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::plus<void>, not_convertible_to_int, iterator_t<test_range<cpp20_input_iterator, int> &> >' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::plus<void>, not_convertible_to_int, cpp20_input_iterator<int *>, invoke_result_t<plus<void> &, not_convertible_to_int, iter_reference_t<cpp20_input_iterator<int *> > > >' evaluated to false}}
    // expected-note@*:* {{because 'convertible_to<not_convertible_to_int, int>' evaluated to false}}
    // expected-note@*:* {{because 'is_convertible_v<not_convertible_to_int, int>' evaluated to false}}
  }
  {
    test_range<cpp20_input_iterator, not_invocable_with_decayed> r;
    std::ranges::fold_left_with_iter(r, 0, std::plus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::plus<void>, int, iterator_t<test_range<cpp20_input_iterator, not_invocable_with_decayed> &> >' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::plus<void>, int, cpp20_input_iterator<not_invocable_with_decayed *>, invoke_result_t<plus<void> &, int, iter_reference_t<cpp20_input_iterator<not_invocable_with_decayed *> > > >' evaluated to false}}
    // expected-note@*:* {{because 'invocable<std::plus<void> &, not_invocable_with_decayed, iter_reference_t<cpp20_input_iterator<not_invocable_with_decayed *> > >' evaluated to false}}
    // expected-note@*:* {{because 'std::invoke(std::forward<_Fn>(__fn), std::forward<_Args>(__args)...)' would be invalid: no matching function for call to 'invoke'}}
  }
  {
    test_range<cpp20_input_iterator, not_assignable_to_decayed> r;
    std::ranges::fold_left_with_iter(r, not_assignable_to_decayed(), std::plus());
    // expected-note@*:* {{because '__indirectly_binary_left_foldable<std::plus<void>, not_assignable_to_decayed, iterator_t<test_range<cpp20_input_iterator, not_assignable_to_decayed> &> >' evaluated to false}}
    // expected-note@*:* {{because '__indirectly_binary_left_foldable_impl<std::plus<void>, not_assignable_to_decayed, cpp20_input_iterator<not_assignable_to_decayed *>, invoke_result_t<plus<void> &, not_assignable_to_decayed, iter_reference_t<cpp20_input_iterator<not_assignable_to_decayed *> > > >' evaluated to false}}
    // expected-note@*:* {{because 'assignable_from<not_assignable_to_decayed &, invoke_result_t<plus<void> &, not_assignable_to_decayed, iter_reference_t<cpp20_input_iterator<not_assignable_to_decayed *> > > >' evaluated to false}}
    // expected-note@*:* {{because '__lhs = std::forward<_Rhs>(__rhs)' would be invalid: no viable overloaded '='}}
  }
}
