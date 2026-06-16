//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// iterator       begin();
// const_iterator begin() const;
// iterator       end();
// const_iterator end() const;

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <valarray>
#include <cassert>
#include <iterator>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template <class T>
void test_return_types() {
  typedef typename std::valarray<T>::iterator It;
  typedef typename std::valarray<T>::const_iterator CIt;

  ASSERT_SAME_TYPE(decltype(std::declval<std::valarray<T> >().begin()), It);
  ASSERT_SAME_TYPE(decltype(std::declval<std::valarray<T>&>().begin()), It);
  ASSERT_SAME_TYPE(decltype(std::declval<std::valarray<T> >().end()), It);
  ASSERT_SAME_TYPE(decltype(std::declval<std::valarray<T>&>().end()), It);

  ASSERT_SAME_TYPE(decltype(std::declval<const std::valarray<T> >().begin()), CIt);
  ASSERT_SAME_TYPE(decltype(std::declval<const std::valarray<T>&>().begin()), CIt);
  ASSERT_SAME_TYPE(decltype(std::declval<const std::valarray<T> >().end()), CIt);
  ASSERT_SAME_TYPE(decltype(std::declval<const std::valarray<T>&>().end()), CIt);
}

int main(int, char**) {
  {
    int a[] = {1, 2, 3, 4, 5};
    std::valarray<int> v(a, 5);
    const std::valarray<int>& cv = v;

    assert(&*v.begin() == &v[0]);
    assert(&*cv.begin() == &cv[0]);
    *v.begin() = 10;
    assert(v[0] == 10);

    assert(&*std::prev(v.end()) == &v[4]);
    assert(&*std::prev(cv.end()) == &cv[4]);
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
  test_return_types<int>();
  test_return_types<double>();

  return 0;
}
