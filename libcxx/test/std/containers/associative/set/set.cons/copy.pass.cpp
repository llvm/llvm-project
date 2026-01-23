//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// set(const set& m);

#include <cassert>
#include <set>

#include "min_allocator.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

template <template <class> class Alloc>
void test_alloc() {
  { // Simple check
    using Set = std::set<int, std::less<int>, Alloc<int> >;

    int arr[] = {1, 2, 3};
    const Set orig(std::begin(arr), std::end(arr));
    Set copy = orig;
    assert(copy.size() == 3);
    assert(*std::next(copy.begin(), 0) == 1);
    assert(*std::next(copy.begin(), 1) == 2);
    assert(*std::next(copy.begin(), 2) == 3);
    assert(std::next(copy.begin(), 3) == copy.end());

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(*std::next(orig.begin(), 0) == 1);
    assert(*std::next(orig.begin(), 1) == 2);
    assert(*std::next(orig.begin(), 2) == 3);
    assert(std::next(orig.begin(), 3) == orig.end());
  }

  { // copy empty set
    using Set = std::set<int, std::less<int>, Alloc<int> >;

    const Set orig;
    Set copy = orig;
    assert(copy.size() == 0);
    assert(copy.begin() == copy.end());

    // Check that orig is still what is expected
    assert(orig.size() == 0);
    assert(orig.begin() == orig.end());
  }

  { // only some leaf nodes exist
    using Set = std::set<int, std::less<int>, Alloc<int> >;

    int arr[] = {1, 2, 3, 4, 5};
    const Set orig(std::begin(arr), std::end(arr));
    Set copy = orig;
    assert(copy.size() == 5);
    assert(*std::next(copy.begin(), 0) == 1);
    assert(*std::next(copy.begin(), 1) == 2);
    assert(*std::next(copy.begin(), 2) == 3);
    assert(*std::next(copy.begin(), 3) == 4);
    assert(*std::next(copy.begin(), 4) == 5);
    assert(std::next(copy.begin(), 5) == copy.end());

    // Check that orig is still what is expected
    assert(orig.size() == 5);
    assert(*std::next(orig.begin(), 0) == 1);
    assert(*std::next(orig.begin(), 1) == 2);
    assert(*std::next(orig.begin(), 2) == 3);
    assert(*std::next(orig.begin(), 3) == 4);
    assert(*std::next(orig.begin(), 4) == 5);
    assert(std::next(orig.begin(), 5) == orig.end());
  }
}

void test() {
  test_alloc<std::allocator>();
  test_alloc<min_allocator>(); // Make sure that fancy pointers work

  { // Ensure that the comparator is copied
    using Set = std::set<int, test_less<int> >;

    int arr[] = {1, 2, 3};
    const Set orig(std::begin(arr), std::end(arr), test_less<int>(3));
    Set copy = orig;
    assert(copy.size() == 3);
    assert(copy.key_comp() == test_less<int>(3));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.key_comp() == test_less<int>(3));
  }

  { // Ensure that the allocator is copied
    using Set = std::set<int, std::less<int>, test_allocator<int> >;

    int arr[] = {1, 2, 3};
    const Set orig(std::begin(arr), std::end(arr), std::less<int>(), test_allocator<int>(10));
    Set copy = orig;
    assert(copy.size() == 3);
    assert(copy.get_allocator() == test_allocator<int>(10));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.get_allocator() == test_allocator<int>(10));
    assert(orig.get_allocator().get_id() != test_alloc_base::moved_value);
  }

  { // Ensure that soccc is handled properly
    using Set = std::set<int, std::less<int>, other_allocator<int> >;

    int arr[] = {1, 2, 3};
    const Set orig(std::begin(arr), std::end(arr), std::less<int>(), other_allocator<int>(10));
    Set copy = orig;
    assert(copy.size() == 3);
    assert(copy.get_allocator() == other_allocator<int>(-2));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.get_allocator() == other_allocator<int>(10));
  }
}

int main(int, char**) {
  test();

  return 0;
}
