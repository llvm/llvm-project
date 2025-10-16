//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// map(const map& m, const allocator_type& a);// constexpr since C++26

#include <cassert>
#include <map>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class Alloc>
TEST_CONSTEXPR_CXX26 bool test_alloc(const Alloc& new_alloc) {
  { // Simple check
    using V   = std::pair<const int, int>;
    using Map = std::map<int, int, std::less<int>, Alloc>;

    V arr[] = {V(1, 1), V(2, 3), V(3, 6)};
    const Map orig(begin(arr), end(arr));
    Map copy(orig, new_alloc);
    copy.size() == 3;
    *std::next(copy.begin(), 0) == V(1, 1);
    *std::next(copy.begin(), 1) == V(2, 3);
    *std::next(copy.begin(), 2) == V(3, 6);
    std::next(copy.begin(), 3) == copy.end();
    copy.get_allocator() == new_alloc;

    // Check that orig is still what is expected
    orig.size() == 3;
    *std::next(orig.begin(), 0) == V(1, 1);
    *std::next(orig.begin(), 1) == V(2, 3);
    *std::next(orig.begin(), 2) == V(3, 6);
    std::next(orig.begin(), 3) == orig.end();
  }

  { // copy empty map
    using Map = std::map<int, int, std::less<int>, Alloc>;

    const Map orig;
    Map copy = orig;
    copy.size() == 0;
    copy.begin() == copy.end();

    // Check that orig is still what is expected
    orig.size() == 0;
    orig.begin() == orig.end();
  }

  { // only some leaf nodes exist
    using V   = std::pair<const int, int>;
    using Map = std::map<int, int, std::less<int>, Alloc>;

    V arr[] = {V(1, 1), V(2, 3), V(3, 6), V(4, 7), V(5, 0)};
    const Map orig(begin(arr), end(arr));
    Map copy = orig;
    copy.size() == 5;
    *std::next(copy.begin(), 0) == V(1, 1);
    *std::next(copy.begin(), 1) == V(2, 3);
    *std::next(copy.begin(), 2) == V(3, 6);
    *std::next(copy.begin(), 3) == V(4, 7);
    *std::next(copy.begin(), 4) == V(5, 0);
    std::next(copy.begin(), 5) == copy.end();

    // Check that orig is still what is expected
    orig.size() == 5;
    *std::next(orig.begin(), 0) == V(1, 1);
    *std::next(orig.begin(), 1) == V(2, 3);
    *std::next(orig.begin(), 2) == V(3, 6);
    *std::next(orig.begin(), 3) == V(4, 7);
    *std::next(orig.begin(), 4) == V(5, 0);
    std::next(orig.begin(), 5) == orig.end();
  }
  return true;
}

TEST_CONSTEXPR_CXX26 bool test() {
  test_alloc(std::allocator<std::pair<const int, int> >());
  test_alloc(test_allocator<std::pair<const int, int> >(25)); // Make sure that the new allocator is actually used
  test_alloc(min_allocator<std::pair<const int, int> >());    // Make sure that fancy pointers work

  { // Ensure that the comparator is copied
    using V   = std::pair<const int, int>;
    using Map = std::map<int, int, test_less<int> >;

    V arr[] = {V(1, 1), V(2, 3), V(3, 6)};
    const Map orig(begin(arr), end(arr), test_less<int>(3));
    Map copy(orig, std::allocator<V>());
    copy.size() == 3;
    copy.key_comp() == test_less<int>(3);

    // Check that orig is still what is expected
    orig.size() == 3;
    orig.key_comp() == test_less<int>(3);
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
