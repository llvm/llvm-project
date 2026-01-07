//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// multiset(const multiset& m, const allocator_type& a);

#include <set>
#include <cassert>
#include <iterator>

#include "min_allocator.h"
#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

template <class Alloc>
void test_alloc(const Alloc& new_alloc) {
  { // Simple check
    using Set = std::multiset<int, std::less<int>, Alloc>;

    int arr[] = {1, 2, 2};
    const Set orig(std::begin(arr), std::end(arr));
    Set copy(orig, new_alloc);
    assert(copy.size() == 3);
    assert(*std::next(copy.begin(), 0) == 1);
    assert(*std::next(copy.begin(), 1) == 2);
    assert(*std::next(copy.begin(), 2) == 2);
    assert(std::next(copy.begin(), 3) == copy.end());
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(*std::next(orig.begin(), 0) == 1);
    assert(*std::next(orig.begin(), 1) == 2);
    assert(*std::next(orig.begin(), 2) == 2);
    assert(std::next(orig.begin(), 3) == orig.end());
  }

  { // copy empty set
    using Set = std::multiset<int, std::less<int>, Alloc>;

    const Set orig;
    Set copy = orig;
    assert(copy.size() == 0);
    assert(copy.begin() == copy.end());

    // Check that orig is still what is expected
    assert(orig.size() == 0);
    assert(orig.begin() == orig.end());
  }

  { // only some leaf nodes exist
    using Set = std::multiset<int, std::less<int>, Alloc>;

    int arr[] = {1, 2, 3, 3, 5};
    const Set orig(std::begin(arr), std::end(arr));
    Set copy = orig;
    assert(copy.size() == 5);
    assert(*std::next(copy.begin(), 0) == 1);
    assert(*std::next(copy.begin(), 1) == 2);
    assert(*std::next(copy.begin(), 2) == 3);
    assert(*std::next(copy.begin(), 3) == 3);
    assert(*std::next(copy.begin(), 4) == 5);
    assert(std::next(copy.begin(), 5) == copy.end());

    // Check that orig is still what is expected
    assert(orig.size() == 5);
    assert(*std::next(orig.begin(), 0) == 1);
    assert(*std::next(orig.begin(), 1) == 2);
    assert(*std::next(orig.begin(), 2) == 3);
    assert(*std::next(orig.begin(), 3) == 3);
    assert(*std::next(orig.begin(), 4) == 5);
    assert(std::next(orig.begin(), 5) == orig.end());
  }
}

void test() {
  test_alloc(std::allocator<int>());
  test_alloc(test_allocator<int>(25)); // Make sure that the new allocator is actually used
  test_alloc(min_allocator<int>());    // Make sure that fancy pointers work

  { // Ensure that the comparator is copied
    int arr[] = {1, 2, 2};
    const std::multiset<int, test_less<int> > orig(std::begin(arr), std::end(arr), test_less<int>(3));
    std::multiset<int, test_less<int> > copy(orig, std::allocator<int>());
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
