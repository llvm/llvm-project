//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// map(const map& m, const allocator_type& a);

#include <cassert>
#include <map>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class Alloc>
void test_alloc(const Alloc& new_alloc) {
  { // Simple check
    using V   = std::pair<const int, int>;
    using Map = std::map<int, int, std::less<int>, Alloc>;

    V arr[] = {V(1, 1), V(2, 3), V(3, 6)};
    const Map orig(begin(arr), end(arr));
    Map copy(orig, new_alloc);
    assert(copy.size() == 3);
    assert(*std::next(copy.begin(), 0) == V(1, 1));
    assert(*std::next(copy.begin(), 1) == V(2, 3));
    assert(*std::next(copy.begin(), 2) == V(3, 6));
    assert(std::next(copy.begin(), 3) == copy.end());
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(*std::next(orig.begin(), 0) == V(1, 1));
    assert(*std::next(orig.begin(), 1) == V(2, 3));
    assert(*std::next(orig.begin(), 2) == V(3, 6));
    assert(std::next(orig.begin(), 3) == orig.end());
  }

  { // copy empty map
    using Map = std::map<int, int, std::less<int>, Alloc>;

    const Map orig;
    Map copy = orig;
    assert(copy.size() == 0);
    assert(copy.begin() == copy.end());

    // Check that orig is still what is expected
    assert(orig.size() == 0);
    assert(orig.begin() == orig.end());
  }

  { // only some leaf nodes exist
    using V   = std::pair<const int, int>;
    using Map = std::map<int, int, std::less<int>, Alloc>;

    V arr[] = {V(1, 1), V(2, 3), V(3, 6), V(4, 7), V(5, 0)};
    const Map orig(begin(arr), end(arr));
    Map copy = orig;
    assert(copy.size() == 5);
    assert(*std::next(copy.begin(), 0) == V(1, 1));
    assert(*std::next(copy.begin(), 1) == V(2, 3));
    assert(*std::next(copy.begin(), 2) == V(3, 6));
    assert(*std::next(copy.begin(), 3) == V(4, 7));
    assert(*std::next(copy.begin(), 4) == V(5, 0));
    assert(std::next(copy.begin(), 5) == copy.end());

    // Check that orig is still what is expected
    assert(orig.size() == 5);
    assert(*std::next(orig.begin(), 0) == V(1, 1));
    assert(*std::next(orig.begin(), 1) == V(2, 3));
    assert(*std::next(orig.begin(), 2) == V(3, 6));
    assert(*std::next(orig.begin(), 3) == V(4, 7));
    assert(*std::next(orig.begin(), 4) == V(5, 0));
    assert(std::next(orig.begin(), 5) == orig.end());
  }
}

void test() {
  test_alloc(std::allocator<std::pair<const int, int> >());
  test_alloc(test_allocator<std::pair<const int, int> >(25)); // Make sure that the new allocator is actually used
  test_alloc(min_allocator<std::pair<const int, int> >());    // Make sure that fancy pointers work

  { // Ensure that the comparator is copied
    using V   = std::pair<const int, int>;
    using Map = std::map<int, int, test_less<int> >;

    V arr[] = {V(1, 1), V(2, 3), V(3, 6)};
    const Map orig(begin(arr), end(arr), test_less<int>(3));
    Map copy(orig, std::allocator<V>());
    assert(copy.size() == 3);
    assert(copy.key_comp() == test_less<int>(3));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.key_comp() == test_less<int>(3));
  }
}

int main(int, char**) {
  test();

  return 0;
}
