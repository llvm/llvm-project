//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <thread>

// class thread::id

// bool operator==(thread::id x, thread::id y) noexcept;
// bool operator!=(thread::id x, thread::id y) noexcept;
// bool operator< (thread::id x, thread::id y) noexcept;
// bool operator<=(thread::id x, thread::id y) noexcept;
// bool operator> (thread::id x, thread::id y) noexcept;
// bool operator>=(thread::id x, thread::id y) noexcept;
// strong_ordering operator<=>(thread::id x, thread::id y) noexcept;

#include <thread>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**) {
  AssertComparisonsAreNoexcept<std::thread::id>();
  AssertComparisonsReturnBool<std::thread::id>();
#if TEST_STD_VER > 17
  AssertOrderAreNoexcept<std::thread::id>();
  AssertOrderReturn<std::strong_ordering, std::thread::id>();
#endif

  std::thread::id id1;
  std::thread::id id2;
  std::thread::id id3 = std::this_thread::get_id();

  // `id1` and `id2` should compare equal
  assert(testComparisons(id1, id2, /*isEqual*/ true, /*isLess*/ false));
#if TEST_STD_VER > 17
  assert(testOrder(id1, id2, std::strong_ordering::equal));
#endif

  // Test `t1` and `t3` which are not equal
  bool isLess = id1 < id3;
  assert(testComparisons(id1, id3, /*isEqual*/ false, isLess));
#if TEST_STD_VER > 17
  assert(testOrder(id1, id3, isLess ? std::strong_ordering::less : std::strong_ordering::greater));
#endif

  // Regression tests for https://github.com/llvm/llvm-project/issues/56187
  // libc++ previously declared the comparison operators as hidden friends
  // which was non-conforming.
  assert(std::operator==(id1, id2));
#if TEST_STD_VER <= 17
  assert(!std::operator!=(id1, id2));
  assert(!std::operator<(id1, id2));
  assert(std::operator<=(id1, id2));
  assert(!std::operator>(id1, id2));
  assert(std::operator>=(id1, id2));
#else
  assert(std::operator<=>(id1, id2) == std::strong_ordering::equal);
#endif

  return 0;
}
