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
//   constexpr bool     // constexpr after c++17
//   equal(Iter1 first1, Iter1 last1, Iter2 first2);
//
// Introduced in C++14:
// template<InputIterator Iter1, InputIterator Iter2>
//   constexpr bool     // constexpr after c++17
//   equal(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter1, class Iter2 = Iter1>
void test_equal() {
  int a[]          = {0, 1, 2, 3, 4, 5};
  const unsigned s = sizeof(a) / sizeof(a[0]);
  int b[s]         = {0, 1, 2, 5, 4, 5};

  assert(std::equal(Iter1(a), Iter1(a + s), Iter2(a)));
  assert(std::equal(Iter2(a), Iter2(a + s), Iter1(a)));
  assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(b)));

#if TEST_STD_VER >= 14
  assert(std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s)));
  assert(std::equal(Iter2(a), Iter2(a + s), Iter1(a), Iter1(a + s)));
  assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s - 1)));
  assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(b), Iter2(b + s)));
#endif
}

#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
    int ia[] = {1, 3, 6, 7};
    int ib[] = {1, 3};
    int ic[] = {1, 3, 5, 7};
    typedef cpp17_input_iterator<int*>         II;
    typedef bidirectional_iterator<int*> BI;

    return !std::equal(std::begin(ia), std::end(ia), std::begin(ic))
        && !std::equal(std::begin(ia), std::end(ia), std::begin(ic), std::end(ic))
        &&  std::equal(std::begin(ib), std::end(ib), std::begin(ic))
        && !std::equal(std::begin(ib), std::end(ib), std::begin(ic), std::end(ic))

        &&  std::equal(II(std::begin(ib)), II(std::end(ib)), II(std::begin(ic)))
        && !std::equal(BI(std::begin(ib)), BI(std::end(ib)), BI(std::begin(ic)), BI(std::end(ic)))
        ;
    }
#endif


int main(int, char**)
{
  test_equal<cpp17_input_iterator<const int*> >();
  test_equal<random_access_iterator<const int*> >();

  // Test all combinations of cv-qualifiers:
  test_equal<int*>();
  test_equal<int*, const int*>();
  test_equal<int*, volatile int*>();
  test_equal<int*, const volatile int*>();
  test_equal<const int*>();
  test_equal<const int*, volatile int*>();
  test_equal<const int*, const volatile int*>();
  test_equal<volatile int*>();
  test_equal<volatile int*, const volatile int*>();
  test_equal<const volatile int*>();

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

  return 0;
}
