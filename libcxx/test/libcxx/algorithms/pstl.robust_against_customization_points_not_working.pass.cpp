//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// Having a customization point outside the module doesn't work, so this test is inherintly module-hostile.
// UNSUPPORTED: clang-modules-build

// Make sure that the customization points get called properly when overloaded

#include <__config>
#include <__iterator/iterator_traits.h>
#include <__iterator/readable_traits.h>
#include <__utility/empty.h>
#include <cassert>
#include <optional>

struct TestPolicy {};
struct TestBackend {};

_LIBCPP_BEGIN_NAMESPACE_STD

bool pstl_any_of_called = false;

template <class, class ForwardIterator, class Pred>
optional<bool> __pstl_any_of(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_any_of_called);
  pstl_any_of_called = true;
  return true;
}

bool pstl_all_of_called = false;

template <class, class ForwardIterator, class Pred>
optional<bool> __pstl_all_of(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_all_of_called);
  pstl_all_of_called = true;
  return true;
}

bool pstl_copy_called = false;

template <class, class ForwardIterator, class ForwardOutIterator>
optional<ForwardOutIterator> __pstl_copy(TestBackend, ForwardIterator, ForwardIterator, ForwardOutIterator res) {
  assert(!pstl_copy_called);
  pstl_copy_called = true;
  return res;
}

bool pstl_copy_n_called = false;

template <class, class ForwardIterator, class Size, class ForwardOutIterator>
optional<ForwardOutIterator> __pstl_copy_n(TestBackend, ForwardIterator, Size, ForwardOutIterator res) {
  assert(!pstl_copy_n_called);
  pstl_copy_n_called = true;
  return res;
}

bool pstl_count_called = false;

template <class, class ForwardIterator, class T>
optional<typename std::iterator_traits<ForwardIterator>::difference_type>
__pstl_count(TestBackend, ForwardIterator, ForwardIterator, const T&) {
  assert(!pstl_count_called);
  pstl_count_called = true;
  return 0;
}

bool pstl_count_if_called = false;

template <class, class ForwardIterator, class Pred>
optional<typename std::iterator_traits<ForwardIterator>::difference_type>
__pstl_count_if(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_count_if_called);
  pstl_count_if_called = true;
  return 0;
}

bool pstl_generate_called = false;

template <class, class ForwardIterator, class Gen>
optional<__empty> __pstl_generate(TestBackend, ForwardIterator, ForwardIterator, Gen) {
  assert(!pstl_generate_called);
  pstl_generate_called = true;
  return __empty{};
}

bool pstl_generate_n_called = false;

template <class, class ForwardIterator, class Size, class Gen>
optional<__empty> __pstl_generate_n(TestBackend, Size, ForwardIterator, Gen) {
  assert(!pstl_generate_n_called);
  pstl_generate_n_called = true;
  return __empty{};
}

bool pstl_none_of_called = false;

template <class, class ForwardIterator, class Pred>
optional<bool> __pstl_none_of(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_none_of_called);
  pstl_none_of_called = true;
  return true;
}

bool pstl_find_called = false;

template <class, class ForwardIterator, class Pred>
optional<ForwardIterator> __pstl_find(TestBackend, ForwardIterator first, ForwardIterator, Pred) {
  assert(!pstl_find_called);
  pstl_find_called = true;
  return first;
}

bool pstl_find_if_called = false;

template <class, class ForwardIterator, class Pred>
optional<ForwardIterator> __pstl_find_if(TestBackend, ForwardIterator first, ForwardIterator, Pred) {
  assert(!pstl_find_if_called);
  pstl_find_if_called = true;
  return first;
}

bool pstl_find_if_not_called = false;

template <class, class ForwardIterator, class Pred>
optional<ForwardIterator> __pstl_find_if_not(TestBackend, ForwardIterator first, ForwardIterator, Pred) {
  assert(!pstl_find_if_not_called);
  pstl_find_if_not_called = true;
  return first;
}

bool pstl_for_each_called = false;

template <class, class ForwardIterator, class Size, class Func>
optional<__empty> __pstl_for_each(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_for_each_called);
  pstl_for_each_called = true;
  return __empty{};
}

bool pstl_for_each_n_called = false;

template <class, class ForwardIterator, class Size, class Func>
optional<__empty> __pstl_for_each_n(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_for_each_n_called);
  pstl_for_each_n_called = true;
  return __empty{};
}

bool pstl_fill_called = false;

template <class, class ForwardIterator, class Size, class Func>
optional<__empty> __pstl_fill(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_fill_called);
  pstl_fill_called = true;
  return __empty{};
}

bool pstl_fill_n_called = false;

template <class, class ForwardIterator, class Size, class Func>
optional<__empty> __pstl_fill_n(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_fill_n_called);
  pstl_fill_n_called = true;
  return __empty{};
}

bool pstl_move_called = false;

template <class, class ForwardIterator, class Size, class Func>
ForwardIterator __pstl_move(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_move_called);
  pstl_move_called = true;
  return 0;
}

bool pstl_is_partitioned_called = false;

template <class, class ForwardIterator, class Func>
optional<bool> __pstl_is_partitioned(TestBackend, ForwardIterator, ForwardIterator, Func) {
  assert(!pstl_is_partitioned_called);
  pstl_is_partitioned_called = true;
  return true;
}

bool pstl_replace_called = false;

template <class, class ForwardIterator, class T>
optional<__empty> __pstl_replace(TestBackend, ForwardIterator, ForwardIterator, const T&, const T&) {
  assert(!pstl_replace_called);
  pstl_replace_called = true;
  return __empty{};
}

bool pstl_replace_if_called = false;

template <class, class ForwardIterator, class T, class Func>
optional<__empty> __pstl_replace_if(TestBackend, ForwardIterator, ForwardIterator, Func, const T&) {
  assert(!pstl_replace_if_called);
  pstl_replace_if_called = true;
  return __empty{};
}

bool pstl_replace_copy_called = false;

template <class, class ForwardIterator, class ForwardOutIterator, class T>
optional<__empty>
__pstl_replace_copy(TestBackend, ForwardIterator, ForwardIterator, ForwardOutIterator, const T&, const T&) {
  assert(!pstl_replace_copy_called);
  pstl_replace_copy_called = true;
  return __empty{};
}

bool pstl_replace_copy_if_called = false;

template <class, class ForwardIterator, class ForwardOutIterator, class T, class Func>
optional<__empty>
__pstl_replace_copy_if(TestBackend, ForwardIterator, ForwardIterator, ForwardOutIterator, Func, const T&) {
  assert(!pstl_replace_copy_if_called);
  pstl_replace_copy_if_called = true;
  return __empty{};
}

bool pstl_unary_transform_called = false;

template <class, class ForwardIterator, class ForwardOutIterator, class UnaryOperation>
optional<ForwardOutIterator>
__pstl_transform(TestBackend, ForwardIterator, ForwardIterator, ForwardOutIterator res, UnaryOperation) {
  assert(!pstl_unary_transform_called);
  pstl_unary_transform_called = true;
  return res;
}

bool pstl_binary_transform_called = false;

template <class, class ForwardIterator1, class ForwardIterator2, class ForwardOutIterator, class BinaryOperation>
optional<ForwardOutIterator> __pstl_transform(
    TestBackend, ForwardIterator1, ForwardIterator1, ForwardIterator2, ForwardOutIterator res, BinaryOperation) {
  assert(!pstl_binary_transform_called);
  pstl_binary_transform_called = true;
  return res;
}

bool pstl_reduce_with_init_called = false;

template <class, class ForwardIterator, class T, class BinaryOperation>
optional<T> __pstl_reduce(TestBackend, ForwardIterator, ForwardIterator, T v, BinaryOperation) {
  assert(!pstl_reduce_with_init_called);
  pstl_reduce_with_init_called = true;
  return v;
}

bool pstl_reduce_without_init_called = false;

template <class, class ForwardIterator>
optional<typename std::iterator_traits<ForwardIterator>::value_type>
__pstl_reduce(TestBackend, ForwardIterator first, ForwardIterator) {
  assert(!pstl_reduce_without_init_called);
  pstl_reduce_without_init_called = true;
  return *first;
}

bool pstl_sort_called = false;

template <class, class RandomAccessIterator, class Comp>
optional<__empty> __pstl_sort(TestBackend, RandomAccessIterator, RandomAccessIterator, Comp) {
  assert(!pstl_sort_called);
  pstl_sort_called = true;
  return __empty{};
}

bool pstl_stable_sort_called = false;

template <class, class RandomAccessIterator, class Comp>
optional<__empty> __pstl_stable_sort(TestBackend, RandomAccessIterator, RandomAccessIterator, Comp) {
  assert(!pstl_stable_sort_called);
  pstl_stable_sort_called = true;
  return __empty{};
}

bool pstl_unary_transform_reduce_called = false;

template <class, class ForwardIterator, class T, class UnaryOperation, class BinaryOperation>
T __pstl_transform_reduce(TestBackend, ForwardIterator, ForwardIterator, T v, UnaryOperation, BinaryOperation) {
  assert(!pstl_unary_transform_reduce_called);
  pstl_unary_transform_reduce_called = true;
  return v;
}

bool pstl_binary_transform_reduce_called = false;

template <class,
          class ForwardIterator1,
          class ForwardIterator2,
          class T,
          class BinaryOperation1,
          class BinaryOperation2>
typename std::iterator_traits<ForwardIterator1>::value_type __pstl_transform_reduce(
    TestBackend, ForwardIterator1, ForwardIterator1, ForwardIterator2, T v, BinaryOperation1, BinaryOperation2) {
  assert(!pstl_binary_transform_reduce_called);
  pstl_binary_transform_reduce_called = true;
  return v;
}

_LIBCPP_END_NAMESPACE_STD

#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>

template <>
inline constexpr bool std::is_execution_policy_v<TestPolicy> = true;

template <>
struct std::__select_backend<TestPolicy> {
  using type = TestBackend;
};

int main(int, char**) {
  int a[]   = {1, 2};
  auto pred = [](auto&&...) { return true; };

  (void)std::any_of(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_any_of_called);
  (void)std::all_of(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_all_of_called);
  (void)std::none_of(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_none_of_called);
  std::copy(TestPolicy{}, std::begin(a), std::end(a), std::begin(a));
  assert(std::pstl_copy_called);
  std::copy_n(TestPolicy{}, std::begin(a), 1, std::begin(a));
  assert(std::pstl_copy_n_called);
  (void)std::count(TestPolicy{}, std::begin(a), std::end(a), 0);
  assert(std::pstl_count_called);
  (void)std::count_if(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_count_if_called);
  (void)std::fill(TestPolicy{}, std::begin(a), std::end(a), 0);
  assert(std::pstl_fill_called);
  (void)std::fill_n(TestPolicy{}, std::begin(a), std::size(a), 0);
  assert(std::pstl_fill_n_called);
  (void)std::find(TestPolicy{}, std::begin(a), std::end(a), 0);
  assert(std::pstl_find_called);
  (void)std::find_if(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_find_if_called);
  (void)std::find_if_not(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_find_if_not_called);
  (void)std::for_each(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_for_each_called);
  (void)std::for_each_n(TestPolicy{}, std::begin(a), std::size(a), pred);
  assert(std::pstl_for_each_n_called);
  (void)std::generate(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_generate_called);
  (void)std::generate_n(TestPolicy{}, std::begin(a), std::size(a), pred);
  assert(std::pstl_generate_n_called);
  (void)std::is_partitioned(TestPolicy{}, std::begin(a), std::end(a), pred);
  assert(std::pstl_is_partitioned_called);
  (void)std::move(TestPolicy{}, std::begin(a), std::end(a), std::begin(a));
  assert(std::pstl_move_called);
  (void)std::replace(TestPolicy{}, std::begin(a), std::end(a), 0, 0);
  assert(std::pstl_replace_called);
  (void)std::replace_if(TestPolicy{}, std::begin(a), std::end(a), pred, 0);
  assert(std::pstl_replace_if_called);
  (void)std::replace_copy(TestPolicy{}, std::begin(a), std::end(a), std::begin(a), 0, 0);
  assert(std::pstl_replace_copy_called);
  (void)std::replace_copy_if(TestPolicy{}, std::begin(a), std::end(a), std::begin(a), pred, 0);
  assert(std::pstl_replace_copy_if_called);
  (void)std::transform(TestPolicy{}, std::begin(a), std::end(a), std::begin(a), pred);
  assert(std::pstl_unary_transform_called);
  (void)std::transform(TestPolicy{}, std::begin(a), std::end(a), std::begin(a), std::begin(a), pred);
  assert(std::pstl_unary_transform_called);
  (void)std::reduce(TestPolicy{}, std::begin(a), std::end(a), 0, pred);
  assert(std::pstl_reduce_with_init_called);
  (void)std::reduce(TestPolicy{}, std::begin(a), std::end(a));
  assert(std::pstl_reduce_without_init_called);
  (void)std::sort(TestPolicy{}, std::begin(a), std::end(a));
  assert(std::pstl_sort_called);
  (void)std::stable_sort(TestPolicy{}, std::begin(a), std::end(a));
  assert(std::pstl_stable_sort_called);
  (void)std::transform_reduce(TestPolicy{}, std::begin(a), std::end(a), 0, pred, pred);
  assert(std::pstl_unary_transform_reduce_called);
  (void)std::transform_reduce(TestPolicy{}, std::begin(a), std::end(a), std::begin(a), 0, pred, pred);
  assert(std::pstl_binary_transform_reduce_called);

  return 0;
}
