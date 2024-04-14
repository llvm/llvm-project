//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   requires EqualityComparable<Iter::value_type>
//   constexpr Iter  // constexpr after C++17
//   adjacent_find(Iter first, Iter last);

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"

struct Test {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    int ia[]          = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(std::adjacent_find(Iter(ia), Iter(ia + sa)) == Iter(ia + 2));
    assert(std::adjacent_find(Iter(ia), Iter(ia)) == Iter(ia));
    assert(std::adjacent_find(Iter(ia + 3), Iter(ia + sa)) == Iter(ia + sa));
  }
};

struct NonTrivial {
  int i_;

  TEST_CONSTEXPR_CXX20 NonTrivial(int i) : i_(i) {}
  TEST_CONSTEXPR_CXX20 NonTrivial(NonTrivial&& other) : i_(other.i_) { other.i_ = 0; }

  TEST_CONSTEXPR_CXX20 friend bool operator==(const NonTrivial& lhs, const NonTrivial& rhs) { return lhs.i_ == rhs.i_; }
};

struct ModTwoComp {
  TEST_CONSTEXPR_CXX20 bool operator()(int lhs, int rhs) { return lhs % 2 == rhs % 2; }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::forward_iterator_list<int*>(), Test());

  { // use a non-integer type to also test the general case - no match
    std::array<NonTrivial, 8> arr = {1, 2, 3, 4, 5, 6, 7, 8};
    assert(std::adjacent_find(arr.begin(), arr.end()) == arr.end());
  }

  { // use a non-integer type to also test the general case - match
    std::array<NonTrivial, 8> lhs = {1, 2, 3, 4, 4, 6, 7, 8};
    assert(std::adjacent_find(lhs.begin(), lhs.end()) == lhs.begin() + 3);
  }

  { // use a custom comparator
    std::array<int, 8> lhs = {0, 1, 2, 3, 5, 6, 7, 8};
    assert(std::adjacent_find(lhs.begin(), lhs.end(), ModTwoComp()) == lhs.begin() + 3);
  }

  return true;
}

template <class T>
void fill_vec(std::vector<T>& vec) {
  for (size_t i = 0; i != vec.size(); ++i) {
    vec[i] = static_cast<T>(i);
  }
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  { // check with a lot of elements to test the vectorization optimization
    {
      std::vector<char> vec(256);
      fill_vec(vec);
      for (size_t i = 0; i != vec.size() - 1; ++i) {
        vec[i] = static_cast<char>(i + 1);
        assert(std::adjacent_find(vec.begin(), vec.end()) == vec.begin() + i);
        vec[i] = static_cast<char>(i);
      }
    }

    {
      std::vector<int> vec(256);
      fill_vec(vec);
      for (size_t i = 0; i != vec.size() - 1; ++i) {
        vec[i] = static_cast<int>(i + 1);
        assert(std::adjacent_find(vec.begin(), vec.end()) == vec.begin() + i);
        vec[i] = static_cast<int>(i);
      }
    }
  }

  { // check the tail of the vectorized loop
    for (size_t vec_size = 2; vec_size != 256; ++vec_size) {
      {
        std::vector<char> vec(vec_size);
        fill_vec(vec);

        assert(std::adjacent_find(vec.begin(), vec.end()) == vec.end());
        vec.back() = static_cast<char>(vec.size() - 2);
        assert(std::adjacent_find(vec.begin(), vec.end()) == vec.end() - 2);
      }
      {
        std::vector<int> vec(vec_size);
        fill_vec(vec);

        assert(std::adjacent_find(vec.begin(), vec.end()) == vec.end());
        vec.back() = static_cast<int>(vec.size() - 2);
        assert(std::adjacent_find(vec.begin(), vec.end()) == vec.end() - 2);
      }
    }
  }

  return 0;
}
