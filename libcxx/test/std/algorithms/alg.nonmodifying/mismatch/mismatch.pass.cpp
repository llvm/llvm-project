//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   constexpr pair<Iter1, Iter2>   // constexpr after c++17
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2);
//
// template<InputIterator Iter1, InputIterator Iter2Pred>
//   constexpr pair<Iter1, Iter2>   // constexpr after c++17
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2); // C++14
//
// template<InputIterator Iter1, InputIterator Iter2,
//          Predicate<auto, Iter1::value_type, Iter2::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr pair<Iter1, Iter2>   // constexpr after c++17
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2, Pred pred);
//
// template<InputIterator Iter1, InputIterator Iter2, Predicate Pred>
//   constexpr pair<Iter1, Iter2>   // constexpr after c++17
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Pred pred); // C++14

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=50000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=100000000

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

template <class Iter, class Container1, class Container2>
TEST_CONSTEXPR_CXX20 void check(Container1 lhs, Container2 rhs, size_t offset) {
  if (lhs.size() == rhs.size()) {
    assert(std::mismatch(Iter(lhs.data()), Iter(lhs.data() + lhs.size()), Iter(rhs.data())) ==
           std::make_pair(Iter(lhs.data() + offset), Iter(rhs.data() + offset)));

    assert(std::mismatch(Iter(lhs.data()),
                         Iter(lhs.data() + lhs.size()),
                         Iter(rhs.data()),
                         std::equal_to<typename Container1::value_type>()) ==
           std::make_pair(Iter(lhs.data() + offset), Iter(rhs.data() + offset)));
  }

#if TEST_STD_VER >= 14
  assert(
      std::mismatch(Iter(lhs.data()), Iter(lhs.data() + lhs.size()), Iter(rhs.data()), Iter(rhs.data() + rhs.size())) ==
      std::make_pair(Iter(lhs.data() + offset), Iter(rhs.data() + offset)));

  assert(std::mismatch(Iter(lhs.data()),
                       Iter(lhs.data() + lhs.size()),
                       Iter(rhs.data()),
                       Iter(rhs.data() + rhs.size()),
                       std::equal_to<typename Container1::value_type>()) ==
         std::make_pair(Iter(lhs.data() + offset), Iter(rhs.data() + offset)));
#endif
}

struct NonTrivial {
  int i_;

  TEST_CONSTEXPR_CXX20 NonTrivial(int i) : i_(i) {}
  TEST_CONSTEXPR_CXX20 NonTrivial(NonTrivial&& other) : i_(other.i_) { other.i_ = 0; }

  TEST_CONSTEXPR_CXX20 friend bool operator==(const NonTrivial& lhs, const NonTrivial& rhs) { return lhs.i_ == rhs.i_; }
};

struct ModTwoComp {
  TEST_CONSTEXPR_CXX20 bool operator()(int lhs, int rhs) { return lhs % 2 == rhs % 2; }
};

template <class Iter>
TEST_CONSTEXPR_CXX20 bool test() {
  { // empty ranges
    std::array<int, 0> lhs = {};
    std::array<int, 0> rhs = {};
    check<Iter>(lhs, rhs, 0);
  }

  { // same range without mismatch
    std::array<int, 8> lhs = {0, 1, 2, 3, 0, 1, 2, 3};
    std::array<int, 8> rhs = {0, 1, 2, 3, 0, 1, 2, 3};
    check<Iter>(lhs, rhs, 8);
  }

  { // same range with mismatch
    std::array<int, 8> lhs = {0, 1, 2, 2, 0, 1, 2, 3};
    std::array<int, 8> rhs = {0, 1, 2, 3, 0, 1, 2, 3};
    check<Iter>(lhs, rhs, 3);
  }

  { // second range is smaller
    std::array<int, 8> lhs = {0, 1, 2, 2, 0, 1, 2, 3};
    std::array<int, 2> rhs = {0, 1};
    check<Iter>(lhs, rhs, 2);
  }

  { // first range is smaller
    std::array<int, 2> lhs = {0, 1};
    std::array<int, 8> rhs = {0, 1, 2, 2, 0, 1, 2, 3};
    check<Iter>(lhs, rhs, 2);
  }

  { // use a custom comparator
    std::array<int, 4> lhs = {0, 2, 3, 4};
    std::array<int, 4> rhs = {0, 0, 4, 4};
    assert(std::mismatch(lhs.data(), lhs.data() + lhs.size(), rhs.data(), ModTwoComp()) ==
           std::make_pair(lhs.data() + 2, rhs.data() + 2));
#if TEST_STD_VER >= 14
    assert(std::mismatch(lhs.data(), lhs.data() + lhs.size(), rhs.data(), rhs.data() + rhs.size(), ModTwoComp()) ==
           std::make_pair(lhs.data() + 2, rhs.data() + 2));
#endif
  }

  return true;
}

struct Test {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    test<Iter>();
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<int*>(), Test());

  { // use a non-integer type to also test the general case - all elements match
    std::array<NonTrivial, 8> lhs = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<NonTrivial, 8> rhs = {1, 2, 3, 4, 5, 6, 7, 8};
    check<NonTrivial*>(std::move(lhs), std::move(rhs), 8);
  }

  { // use a non-integer type to also test the general case - not all elements match
    std::array<NonTrivial, 8> lhs = {1, 2, 3, 4, 7, 6, 7, 8};
    std::array<NonTrivial, 8> rhs = {1, 2, 3, 4, 5, 6, 7, 8};
    check<NonTrivial*>(std::move(lhs), std::move(rhs), 4);
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  { // check with a lot of elements to test the vectorization optimization
    {
      std::vector<char> lhs(256);
      std::vector<char> rhs(256);
      for (size_t i = 0; i != lhs.size(); ++i) {
        lhs[i] = 1;
        check<char*>(lhs, rhs, i);
        lhs[i] = 0;
        rhs[i] = 1;
        check<char*>(lhs, rhs, i);
        rhs[i] = 0;
      }
    }

    {
      std::vector<int> lhs(256);
      std::vector<int> rhs(256);
      for (size_t i = 0; i != lhs.size(); ++i) {
        lhs[i] = 1;
        check<int*>(lhs, rhs, i);
        lhs[i] = 0;
        rhs[i] = 1;
        check<int*>(lhs, rhs, i);
        rhs[i] = 0;
      }
    }
  }

  return 0;
}
