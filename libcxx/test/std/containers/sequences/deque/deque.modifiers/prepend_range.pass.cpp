//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// template<container-compatible-range<T> R>
//   constexpr void prepend_range(R&& rg); // C++23

#include <deque>

#include "../../insert_range_sequence_containers.h"
#include "test_macros.h"

#if !defined(TEST_HAS_NO_EXCEPTIONS)
struct ThrowingSwap {
  static bool throwing_enabled;
  static int swaps;

  int value = 0;

  ThrowingSwap() = default;
  ThrowingSwap(int v) : value(v) {}

  friend void swap(ThrowingSwap& lhs, ThrowingSwap& rhs) {
    int tmp   = lhs.value;
    lhs.value = rhs.value;
    rhs.value = tmp;
    if (throwing_enabled && ++swaps == 2)
      throw 1;
  }

  friend bool operator==(const ThrowingSwap& lhs, const ThrowingSwap& rhs) { return lhs.value == rhs.value; }
};

bool ThrowingSwap::throwing_enabled = false;
int ThrowingSwap::swaps             = 0;
#endif

void test_input_range_strong_exception_safety() {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  using T = ThrowingCopy<3>;

  T::throwing_enabled = false;
  std::deque<T> c;
  for (int i = 0; i != 8; ++i)
    c.emplace_back(100 + i);
  for (int i = 0; i != 4; ++i)
    c.pop_front();
  std::deque<T> expected = c;

  T input[] = {T(0), T(1), T(2), T(3)};
  using Iter = cpp20_input_iterator<T*>;
  auto in    = std::ranges::subrange(Iter(input), sentinel_wrapper<Iter>(Iter(input + 4)));

  T::reset();
  T::throwing_enabled = true;
  try {
    c.prepend_range(in);
    assert(false);
  } catch (int) {
    assert(c == expected);
  }
  T::throwing_enabled = false;
#endif
}

void test_input_range_reverse_exception_safety() {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  std::deque<ThrowingSwap> c = {100, 101, 102, 103, 104, 105, 106, 107};
  for (int i = 0; i != 4; ++i)
    c.pop_front();
  std::deque<ThrowingSwap> expected = c;

  ThrowingSwap input[] = {0, 1, 2, 3};
  using Iter           = cpp20_input_iterator<ThrowingSwap*>;
  auto in              = std::ranges::subrange(Iter(input), sentinel_wrapper<Iter>(Iter(input + 4)));

  ThrowingSwap::swaps            = 0;
  ThrowingSwap::throwing_enabled = true;
  try {
    c.prepend_range(in);
    assert(false);
  } catch (int) {
    assert(c == expected);
  }
  ThrowingSwap::throwing_enabled = false;
#endif
}

// Tested cases:
// - different kinds of insertions (prepending an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container);
// - prepending move-only elements;
// - an exception is thrown when copying the elements or when allocating new elements.
int main(int, char**) {
  static_assert(test_constraints_prepend_range<std::deque, int, double>());

  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_prepend_range<std::deque<int, Alloc>, Iter, Sent>([]([[maybe_unused]] auto&& c) {
      LIBCPP_ASSERT(c.__invariants());
    });
  });
  test_sequence_prepend_range_move_only<std::deque>();
  // FIXME: This should work - see https://llvm.org/PR162605
  // test_sequence_prepend_range_emplace_constructible<std::deque>();

  test_prepend_range_exception_safety_throwing_copy<std::deque>();
  test_prepend_range_exception_safety_throwing_allocator<std::deque, int>();
  test_input_range_strong_exception_safety();
  test_input_range_reverse_exception_safety();

  return 0;
}
