//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// Ensure that all the elements in the inplace_vector are destroyed

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>
#include <inplace_vector>

#include "test_macros.h"

struct DestroyTracker {
  constexpr DestroyTracker(std::vector<bool>& vec) : vec_(&vec), index_(vec.size()) { vec.push_back(false); }

  constexpr DestroyTracker(const DestroyTracker& other) : vec_(other.vec_), index_(vec_->size()) {
    vec_->push_back(false);
  }

  constexpr DestroyTracker& operator=(const DestroyTracker&) { return *this; }
  constexpr ~DestroyTracker() { (*vec_)[index_] = true; }

  std::vector<bool>* vec_;
  size_t index_;
};

template <std::size_t N, class Operation>
constexpr void test(Operation operation) {
  std::vector<bool> all_destroyed;

  {
    std::inplace_vector<DestroyTracker, N> v;
    for (size_t i = 0; i != 100; ++i)
      operation(v, all_destroyed);
  }

  assert(std::all_of(all_destroyed.begin(), all_destroyed.end(), [](bool b) { return b; }));
}

template <typename V>
constexpr void checked_op(V& vec, std::vector<bool>& tracker, int operation_number) {
  std::size_t needed = (operation_number < 0) ? 2 : 1;
  bool has_capacity  = vec.capacity() - vec.size() >= needed;
  if consteval {
    if (!has_capacity)
      return; // No exceptions in constant expression
  }
#ifdef TEST_HAS_NO_EXCEPTIONS
  if (!has_capacity)
    return;
#else
  try {
#endif
  switch (operation_number) {
  case 0:
    vec.emplace_back(tracker);
    break;
  case 1:
    vec.push_back(DestroyTracker(tracker));
    break;
  case 2:
    vec.push_back(static_cast<const DestroyTracker&>(DestroyTracker(tracker)));
    break;
  case 3:
    vec.emplace(vec.begin(), tracker);
    break;
  case 4:
    vec.insert(vec.begin(), DestroyTracker(tracker));
    break;
  case 5:
    vec.insert(vec.begin(), static_cast<const DestroyTracker&>(DestroyTracker(tracker)));
    break;
  case -1:
    vec.insert_range(vec.begin(), std::array<DestroyTracker, 2>{tracker, tracker});
    break;
  case -2:
    vec.append_range(std::array<DestroyTracker, 2>{tracker, tracker});
    break;
  }
  assert(has_capacity);
#ifndef TEST_HAS_NO_EXCEPTIONS
}
catch (const std::bad_alloc&) {
  assert(!has_capacity);
}
catch (...) {
  assert(false);
}
#endif
}

template <typename V>
constexpr void single_try_op(V& vec, std::vector<bool>& tracker, int operation_number) {
  std::size_t needed = (operation_number < 0) ? 2 : 1;
  bool has_capacity  = vec.capacity() - vec.size() >= needed;

  switch (operation_number) {
  case 0:
    vec.try_emplace_back(tracker);
    break;
  case 1:
    vec.try_push_back(DestroyTracker(tracker));
    break;
  case 2:
    vec.try_push_back(static_cast<const DestroyTracker&>(DestroyTracker(tracker)));
    break;
  case 3:
    vec.try_emplace(vec.begin(), tracker);
    break;
  case 4:
    vec.try_insert(vec.begin(), DestroyTracker(tracker));
    break;
  case 5:
    vec.try_insert(vec.begin(), static_cast<const DestroyTracker&>(DestroyTracker(tracker)));
    break;
  case -1:
    vec.insert_range(vec.begin(), std::array<DestroyTracker, 2>{tracker, tracker});
    break;
  case -2:
    vec.append_range(std::array<DestroyTracker, 2>{tracker, tracker});
    break;
  }
  assert(has_capacity);
}

template <std::size_t N>
constexpr bool test() {
  using V = std::inplace_vector<DestroyTracker, N>;
  for (int i = -2; i <= 5; ++i) {
    test<N>([i](V& vec, std::vector<bool>& tracker) { checked_op(vec, tracker, i); });
  }

  return true;
}

constexpr bool tests() {
  test<0>();
  test<10>();
  test<100>();
  return true;
}

int main() {
  tests();
  // inplace_vector does not support constexpr evaluation when type is not trivially destructible
  // static_assert(tests());
}
