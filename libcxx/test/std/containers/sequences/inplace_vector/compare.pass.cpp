//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class T, size_t N>
//   constexpr bool operator==(const inplace_vector<T, N>& x, const inplace_vector<T, N>& y);

#include <cassert>
#include <concepts>
#include <inplace_vector>

#include "common.h"
#include "test_comparisons.h"

constexpr bool test() {
  {
    std::inplace_vector<int, 8> c1;
    std::inplace_vector<int, 8> c2;
    std::same_as<bool> decltype(auto) result = c1 == c2;
    assert(result);
    assert(!(c1 != c2));
  }
  {
    std::inplace_vector<int, 8> c1{1, 2, 3};
    std::inplace_vector<int, 8> c2{1, 2, 3};
    std::inplace_vector<int, 8> c3{1, 2, 4};
    std::inplace_vector<int, 8> c4{1, 2};
    assert(c1 == c2);
    assert(c1 != c3);
    assert(c1 != c4);
  }
  { // all six comparison operators (!=, <, <=, >, >= are rewritten in terms of == and <=>)
    std::inplace_vector<int, 8> c1;
    std::inplace_vector<int, 8> c2;
    assert(testComparisons(c1, c2, true, false));
  }
  {
    std::inplace_vector<int, 8> c1{1, 2, 1};
    std::inplace_vector<int, 8> c2{1, 2, 2};
    assert(testComparisons(c1, c2, false, true));
  }
  {
    std::inplace_vector<int, 8> c1{3, 2, 3};
    std::inplace_vector<int, 8> c2{3, 1, 3};
    assert(testComparisons(c1, c2, false, false));
  }
  { // a shorter inplace_vector that is a prefix of the longer one compares less
    std::inplace_vector<int, 8> c1{1, 2};
    std::inplace_vector<int, 8> c2{1, 2, 0};
    assert(testComparisons(c1, c2, false, true));
  }
  {
    std::inplace_vector<int, 8> c1{1, 2, 0};
    std::inplace_vector<int, 8> c2{3};
    assert(testComparisons(c1, c2, false, true));
  }

  // types that only provide operator< and operator==
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    {
      std::inplace_vector<LessAndEqComp, 8> c1;
      std::inplace_vector<LessAndEqComp, 8> c2;
      assert(testComparisons(c1, c2, true, false));
    }
    {
      std::inplace_vector<LessAndEqComp, 8> c1{LessAndEqComp(1), LessAndEqComp(2)};
      std::inplace_vector<LessAndEqComp, 8> c2{LessAndEqComp(1), LessAndEqComp(2)};
      assert(testComparisons(c1, c2, true, false));
    }
    {
      std::inplace_vector<LessAndEqComp, 8> c1{LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(2)};
      std::inplace_vector<LessAndEqComp, 8> c2{LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(1)};
      assert(testComparisons(c1, c2, false, false));
    }
    {
      std::inplace_vector<LessAndEqComp, 8> c1{LessAndEqComp(1), LessAndEqComp(2)};
      std::inplace_vector<LessAndEqComp, 8> c2{LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(0)};
      assert(testComparisons(c1, c2, false, true));
    }
  }

  { // zero capacity inplace_vectors are always equal
    std::inplace_vector<int, 0> c1;
    std::inplace_vector<int, 0> c2;
    assert(testComparisons(c1, c2, true, false));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
