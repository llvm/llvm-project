//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// template <class InputIterator>
//   void insert(InputIterator first, InputIterator last);

#include <array>
#include <cassert>
#include <map>

#include "test_iterators.h"
#include "min_allocator.h"

template <class Iter, class Alloc>
void test_alloc() {
  {   // Check that an empty range works correctly
    { // Without elements in the container
      using Map = std::map<int, int, std::less<int>, Alloc>;

      std::array<std::pair<const int, int>, 0> arr;

      Map map;
      map.insert(Iter(arr.data()), Iter(arr.data() + arr.size()));
      assert(map.size() == 0);
      assert(map.begin() == map.end());
    }
    { // With 1 element in the container
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      std::array<Pair, 0> arr;

      Map map;
      map.insert(Pair(0, 0));
      map.insert(Iter(arr.data()), Iter(arr.data() + arr.size()));
      assert(map.size() == 1);
      assert(*std::next(map.begin(), 0) == Pair(0, 0));
      assert(std::next(map.begin(), 1) == map.end());
    }
    { // With multiple elements in the container
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      std::array<Pair, 0> arr;

      Map map;
      map.insert(Pair(0, 0));
      map.insert(Pair(1, 1));
      map.insert(Pair(2, 2));
      map.insert(Iter(arr.data()), Iter(arr.data() + arr.size()));
      assert(map.size() == 3);
      assert(*std::next(map.begin(), 0) == Pair(0, 0));
      assert(*std::next(map.begin(), 1) == Pair(1, 1));
      assert(*std::next(map.begin(), 2) == Pair(2, 2));
      assert(std::next(map.begin(), 3) == map.end());
    }
  }
  {   // Check that 1 element is inserted correctly
    { // Without elements in the container
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      Pair arr[] = {Pair(1, 1)};

      Map map;
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 1);
      assert(*std::next(map.begin(), 0) == Pair(1, 1));
      assert(std::next(map.begin(), 1) == map.end());
    }
    { // With 1 element in the container - a different key
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      Pair arr[] = {Pair(1, 1)};

      Map map;
      map.insert(Pair(0, 0));
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 2);
      assert(*std::next(map.begin(), 0) == Pair(0, 0));
      assert(*std::next(map.begin(), 1) == Pair(1, 1));
      assert(std::next(map.begin(), 2) == map.end());
    }
    { // With 1 element in the container - the same key
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      Pair arr[] = {Pair(1, 1)};

      Map map;
      map.insert(Pair(1, 2));
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 1);
      assert(*std::next(map.begin(), 0) == Pair(1, 2));
      assert(std::next(map.begin(), 1) == map.end());
    }
    { // With multiple elements in the container
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      Pair arr[] = {Pair(1, 1)};

      Map map;
      map.insert(Pair(0, 0));
      map.insert(Pair(1, 1));
      map.insert(Pair(2, 2));
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 3);
      assert(*std::next(map.begin(), 0) == Pair(0, 0));
      assert(*std::next(map.begin(), 1) == Pair(1, 1));
      assert(*std::next(map.begin(), 2) == Pair(2, 2));
      assert(std::next(map.begin(), 3) == map.end());
    }
  }
  {   // Check that multiple elements are inserted correctly
    { // Without elements in the container
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      Pair arr[] = {Pair(1, 1), Pair(1, 1), Pair(3, 3)};

      Map map;
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 2);
      assert(*std::next(map.begin(), 0) == Pair(1, 1));
      assert(*std::next(map.begin(), 1) == Pair(3, 3));
      assert(std::next(map.begin(), 2) == map.end());
    }
    { // With 1 element in the container - a different key
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      Pair arr[] = {Pair(1, 1), Pair(1, 1), Pair(3, 3)};

      Map map;
      map.insert(Pair(0, 0));
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 3);
      assert(*std::next(map.begin(), 0) == Pair(0, 0));
      assert(*std::next(map.begin(), 1) == Pair(1, 1));
      assert(*std::next(map.begin(), 2) == Pair(3, 3));
      assert(std::next(map.begin(), 3) == map.end());
    }
    { // With 1 element in the container - the same key
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      Pair arr[] = {Pair(1, 1), Pair(2, 2), Pair(3, 3)};

      Map map;
      map.insert(Pair(1, 1));
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 3);
      assert(*std::next(map.begin(), 0) == Pair(1, 1));
      assert(*std::next(map.begin(), 1) == Pair(2, 2));
      assert(*std::next(map.begin(), 2) == Pair(3, 3));
      assert(std::next(map.begin(), 3) == map.end());
    }
    { // With multiple elements in the container
      using Map  = std::map<int, int, std::less<int>, Alloc>;
      using Pair = std::pair<const int, int>;

      Pair arr[] = {Pair(1, 1), Pair(3, 3), Pair(4, 4)};

      Map map;
      map.insert(Pair(0, 0));
      map.insert(Pair(1, 1));
      map.insert(Pair(2, 2));
      map.insert(Iter(std::begin(arr)), Iter(std::end(arr)));
      assert(map.size() == 5);
      assert(*std::next(map.begin(), 0) == Pair(0, 0));
      assert(*std::next(map.begin(), 1) == Pair(1, 1));
      assert(*std::next(map.begin(), 2) == Pair(2, 2));
      assert(*std::next(map.begin(), 3) == Pair(3, 3));
      assert(*std::next(map.begin(), 4) == Pair(4, 4));
      assert(std::next(map.begin(), 5) == map.end());
    }
  }
}

void test() {
  test_alloc<cpp17_input_iterator<std::pair<const int, int>*>, std::allocator<std::pair<const int, int> > >();
  test_alloc<cpp17_input_iterator<std::pair<const int, int>*>, min_allocator<std::pair<const int, int> > >();
}

int main(int, char**) {
  test();

  return 0;
}
