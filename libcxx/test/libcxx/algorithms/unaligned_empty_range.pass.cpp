//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Some algorithms make alignment assumptions for optimization purposes. However, it is
// fairly common for data structures to implement iterators for empty ranges using
// sentinel values, which may not be properly aligned. This test ensures that we don't
// make alignment assumptions for empty ranges, which would break that use case.

// UNSUPPORTED: c++03

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ranges>

#include "test_macros.h"

// Returns a pointer that doesn't point to any object at all and that is not suitably
// aligned for T. This is the sort of sentinel value that user code sometimes uses to
// represent an empty range.
template <class T>
T* unaligned_sentinel() {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(1));
}

template <class T>
void test() {
  T* first = unaligned_sentinel<T>();
  T* last  = first;
  T value{};
  auto pred = [](T const&) { return true; };

  assert(std::find(first, last, value) == last);
  assert(std::find_if(first, last, pred) == last);
  assert(std::find_if_not(first, last, pred) == last);
  assert(std::any_of(first, last, pred) == false);
  assert(std::all_of(first, last, pred) == true);
  assert(std::none_of(first, last, pred) == true);
  assert(std::remove(first, last, value) == last);
  assert(std::remove_if(first, last, pred) == last);

#if TEST_STD_VER >= 20
  // (iterator, sentinel) overloads
  assert(std::ranges::find(first, last, value) == last);
  assert(std::ranges::find_if(first, last, pred) == last);
  assert(std::ranges::find_if_not(first, last, pred) == last);
  assert(std::ranges::any_of(first, last, pred) == false);
  assert(std::ranges::all_of(first, last, pred) == true);
  assert(std::ranges::none_of(first, last, pred) == true);
  assert(std::ranges::remove(first, last, value).begin() == last);
  assert(std::ranges::remove_if(first, last, pred).begin() == last);

  // (range) overloads
  std::ranges::subrange range(first, last);
  assert(std::ranges::find(range, value) == last);
  assert(std::ranges::find_if(range, pred) == last);
  assert(std::ranges::find_if_not(range, pred) == last);
  assert(std::ranges::any_of(range, pred) == false);
  assert(std::ranges::all_of(range, pred) == true);
  assert(std::ranges::none_of(range, pred) == true);
  assert(std::ranges::remove(range, value).begin() == last);
  assert(std::ranges::remove_if(range, pred).begin() == last);
#endif
}

struct alignas(16) Overaligned {
  int value = 0;
  friend bool operator==(Overaligned const& x, Overaligned const& y) { return x.value == y.value; }
};

int main(int, char**) {
  test<int>();
  test<double>(); // not trivially equality comparable
  test<Overaligned>();

  return 0;
}
