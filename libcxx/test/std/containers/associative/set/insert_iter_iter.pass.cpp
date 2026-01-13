//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// template <class InputIterator>
//   void insert(InputIterator first, InputIterator last);

#include <array>
#include <cassert>
#include <set>

#include "min_allocator.h"
#include "test_iterators.h"

template <class Iter, class Alloc>
void test_alloc() {
  {   // Check that an empty range works correctly
    { // Without elements in the container
      using Map = std::set<int, std::less<int>, Alloc>;

      std::array<int, 0> arr;

      Map map;
      map.insert(Iter(arr.data()), Iter(arr.data() + arr.size()));
      assert(map.size() == 0);
      assert(map.begin() == map.end());
    }
    { // With 1 element in the container
      using Map = std::set<int, std::less<int>, Alloc>;

      std::array<int, 0> arr;

      Map map;
      map.insert(0);
      map.insert(Iter(arr.data()), Iter(arr.data() + arr.size()));
      assert(map.size() == 1);
      assert(*std::next(map.begin(), 0) == 0);
      assert(std::next(map.begin(), 1) == map.end());
    }
    { // With multiple elements in the container
      using Map = std::set<int, std::less<int>, Alloc>;

      std::array<int, 0> arr;

      Map map;
      map.insert(0);
      map.insert(1);
      map.insert(2);
      map.insert(Iter(arr.data()), Iter(arr.data() + arr.size()));
      assert(map.size() == 3);
      assert(*std::next(map.begin(), 0) == 0);
      assert(*std::next(map.begin(), 1) == 1);
      assert(*std::next(map.begin(), 2) == 2);
      assert(std::next(map.begin(), 3) == map.end());
    }
  }
  {   // Check that 1 element is inserted correctly
    { // Without elements in the container
      using Map = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1};

      Map map;
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 1);
      assert(*std::next(map.begin(), 0) == 1);
      assert(std::next(map.begin(), 1) == map.end());
    }
    { // With 1 element in the container - a different key
      using Map = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1};

      Map map;
      map.insert(0);
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 2);
      assert(*std::next(map.begin(), 0) == 0);
      assert(*std::next(map.begin(), 1) == 1);
      assert(std::next(map.begin(), 2) == map.end());
    }
    { // With 1 element in the container - the same key
      using Map = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1};

      Map map;
      map.insert(1);
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 1);
      assert(*std::next(map.begin(), 0) == 1);
      assert(std::next(map.begin(), 1) == map.end());
    }
    { // With multiple elements in the container
      using Map = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1};

      Map map;
      map.insert(0);
      map.insert(1);
      map.insert(2);
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 3);
      assert(*std::next(map.begin(), 0) == 0);
      assert(*std::next(map.begin(), 1) == 1);
      assert(*std::next(map.begin(), 2) == 2);
      assert(std::next(map.begin(), 3) == map.end());
    }
  }
  {   // Check that multiple elements are inserted correctly
    { // Without elements in the container
      using Map = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1, 1, 3};

      Map map;
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 2);
      assert(*std::next(map.begin(), 0) == 1);
      assert(*std::next(map.begin(), 1) == 3);
      assert(std::next(map.begin(), 2) == map.end());
    }
    { // With 1 element in the container - a different key
      using Map = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1, 1, 3};

      Map map;
      map.insert(0);
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 3);
      assert(*std::next(map.begin(), 0) == 0);
      assert(*std::next(map.begin(), 1) == 1);
      assert(*std::next(map.begin(), 2) == 3);
      assert(std::next(map.begin(), 3) == map.end());
    }
    { // With 1 element in the container - the same key
      using Map = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1, 2, 3};

      Map map;
      map.insert(1);
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 3);
      assert(*std::next(map.begin(), 0) == 1);
      assert(*std::next(map.begin(), 1) == 2);
      assert(*std::next(map.begin(), 2) == 3);
      assert(std::next(map.begin(), 3) == map.end());
    }
    { // With multiple elements in the container
      using Map = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1, 3, 4};

      Map map;
      map.insert(0);
      map.insert(1);
      map.insert(2);
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 5);
      assert(*std::next(map.begin(), 0) == 0);
      assert(*std::next(map.begin(), 1) == 1);
      assert(*std::next(map.begin(), 2) == 2);
      assert(*std::next(map.begin(), 3) == 3);
      assert(*std::next(map.begin(), 4) == 4);
      assert(std::next(map.begin(), 5) == map.end());
    }
  }
}

void test() {
  test_alloc<cpp17_input_iterator<int*>, std::allocator<int> >();
  test_alloc<cpp17_input_iterator<int*>, min_allocator<int> >();
}

int main(int, char**) {
  test();

  return 0;
}
