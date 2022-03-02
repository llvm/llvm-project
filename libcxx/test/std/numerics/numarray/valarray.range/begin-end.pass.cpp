//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// template <class T> unspecified begin(valarray<T>& v);
// template <class T> unspecified begin(const valarray<T>& v);
// template <class T> unspecified end(valarray<T>& v);
// template <class T> unspecified end(const valarray<T>& v);

#include <valarray>
#include <cassert>
#include <iterator>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
  {
    int a[] = {1, 2, 3, 4, 5};
    std::valarray<int> v(a, 5);
    const std::valarray<int>& cv = v;
    using It = decltype(std::begin(v));
    using CIt = decltype(std::begin(cv));
    static_assert(std::is_base_of<std::random_access_iterator_tag, std::iterator_traits<It>::iterator_category>::value, "");
    static_assert(std::is_base_of<std::random_access_iterator_tag, std::iterator_traits<CIt>::iterator_category>::value, "");
    ASSERT_SAME_TYPE(decltype(*std::begin(v)), int&);
    ASSERT_SAME_TYPE(decltype(*std::begin(cv)), const int&);
    assert(&*std::begin(v) == &v[0]);
    assert(&*std::begin(cv) == &cv[0]);
    *std::begin(v) = 10;
    assert(v[0] == 10);

    ASSERT_SAME_TYPE(decltype(std::end(v)), It);
    ASSERT_SAME_TYPE(decltype(std::end(cv)), CIt);
    assert(&*std::prev(std::end(v)) == &v[4]);
    assert(&*std::prev(std::end(cv)) == &cv[4]);
  }
#if TEST_STD_VER >= 11
  {
    int a[] = {1, 2, 3, 4, 5};
    std::valarray<int> v(a, 5);
    int sum = 0;
    for (int& i : v) {
      sum += i;
    }
    assert(sum == 15);
  }
  {
    int a[] = {1, 2, 3, 4, 5};
    const std::valarray<int> cv(a, 5);
    int sum = 0;
    for (const int& i : cv) {
      sum += i;
    }
    assert(sum == 15);
  }
#endif

  return 0;
}
