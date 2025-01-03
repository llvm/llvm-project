//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2>
//   requires HasLess<Iter1::value_type, Iter2::value_type>
//         && HasLess<Iter2::value_type, Iter1::value_type>
//   constexpr bool             // constexpr after C++17
//   lexicographical_compare(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class T, class Iter1>
struct Test {
  template <class Iter2>
  TEST_CONSTEXPR_CXX20 void operator()() {
    T ia[]            = {1, 2, 3, 4};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    T ib[]            = {1, 2, 3};
    assert(!std::lexicographical_compare(Iter1(ia), Iter1(ia + sa), Iter2(ib), Iter2(ib + 2)));
    assert(std::lexicographical_compare(Iter1(ib), Iter1(ib + 2), Iter2(ia), Iter2(ia + sa)));
    assert(!std::lexicographical_compare(Iter1(ia), Iter1(ia + sa), Iter2(ib), Iter2(ib + 3)));
    assert(std::lexicographical_compare(Iter1(ib), Iter1(ib + 3), Iter2(ia), Iter2(ia + sa)));
    assert(std::lexicographical_compare(Iter1(ia), Iter1(ia + sa), Iter2(ib + 1), Iter2(ib + 3)));
    assert(!std::lexicographical_compare(Iter1(ib + 1), Iter1(ib + 3), Iter2(ia), Iter2(ia + sa)));
  }
};

template <class T>
struct TestIter {
  template <class Iter1>
  TEST_CONSTEXPR_CXX20 bool operator()() {
    types::for_each(types::cpp17_input_iterator_list<T*>(), Test<T, Iter1>());

    return true;
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<const int*>(), TestIter<const int>());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  types::for_each(types::cpp17_input_iterator_list<const wchar_t*>(), TestIter<const wchar_t>());
#endif
  types::for_each(types::cpp17_input_iterator_list<const char*>(), TestIter<const char>());
  types::for_each(types::cpp17_input_iterator_list<unsigned char*>(), TestIter<unsigned char>());

  return true;
}

int main(int, char**) {
  test();

#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
