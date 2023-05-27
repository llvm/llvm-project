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
// UNSUPPORTED: modules-build

// Make sure that the customization points get called properly when overloaded

#include <__config>
#include <cassert>

struct TestPolicy {};
struct TestBackend {};

_LIBCPP_BEGIN_NAMESPACE_STD

bool pstl_any_of_called = false;

template <class, class ForwardIterator, class Pred>
bool __pstl_any_of(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_any_of_called);
  pstl_any_of_called = true;
  return true;
}

bool pstl_all_of_called = false;

template <class, class ForwardIterator, class Pred>
bool __pstl_all_of(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_all_of_called);
  pstl_all_of_called = true;
  return true;
}

bool pstl_none_of_called = false;

template <class, class ForwardIterator, class Pred>
bool __pstl_none_of(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_none_of_called);
  pstl_none_of_called = true;
  return true;
}

bool pstl_find_called = false;

template <class, class ForwardIterator, class Pred>
ForwardIterator __pstl_find(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_find_called);
  pstl_find_called = true;
  return {};
}

bool pstl_find_if_called = false;

template <class, class ForwardIterator, class Pred>
ForwardIterator __pstl_find_if(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_find_if_called);
  pstl_find_if_called = true;
  return {};
}

bool pstl_find_if_not_called = false;

template <class, class ForwardIterator, class Pred>
ForwardIterator __pstl_find_if_not(TestBackend, ForwardIterator, ForwardIterator, Pred) {
  assert(!pstl_find_if_not_called);
  pstl_find_if_not_called = true;
  return {};
}

bool pstl_for_each_called = false;

template <class, class ForwardIterator, class Size, class Func>
void __pstl_for_each(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_for_each_called);
  pstl_for_each_called = true;
}

bool pstl_for_each_n_called = false;

template <class, class ForwardIterator, class Size, class Func>
void __pstl_for_each_n(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_for_each_n_called);
  pstl_for_each_n_called = true;
}

bool pstl_fill_called = false;

template <class, class ForwardIterator, class Size, class Func>
void __pstl_fill(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_fill_called);
  pstl_fill_called = true;
}

bool pstl_fill_n_called = false;

template <class, class ForwardIterator, class Size, class Func>
void __pstl_fill_n(TestBackend, ForwardIterator, Size, Func) {
  assert(!pstl_fill_n_called);
  pstl_fill_n_called = true;
}

bool pstl_unary_transform_called = false;

template <class, class ForwardIterator, class ForwardOutIterator, class UnaryOperation>
ForwardOutIterator __pstl_transform(TestBackend, ForwardIterator, ForwardIterator, ForwardOutIterator, UnaryOperation) {
  assert(!pstl_unary_transform_called);
  pstl_unary_transform_called = true;
  return {};
}

bool pstl_binary_transform_called = false;

template <class, class ForwardIterator1, class ForwardIterator2, class ForwardOutIterator, class BinaryOperation>
ForwardOutIterator __pstl_transform(
    TestBackend, ForwardIterator1, ForwardIterator1, ForwardIterator2, ForwardOutIterator, BinaryOperation) {
  assert(!pstl_binary_transform_called);
  pstl_binary_transform_called = true;
  return {};
}

_LIBCPP_END_NAMESPACE_STD

#include <algorithm>
#include <cassert>
#include <iterator>

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
  (void)std::transform(TestPolicy{}, std::begin(a), std::end(a), std::begin(a), pred);
  assert(std::pstl_unary_transform_called);
  (void)std::transform(TestPolicy{}, std::begin(a), std::end(a), std::begin(a), std::begin(a), pred);
  assert(std::pstl_unary_transform_called);

  return 0;
}
