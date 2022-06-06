//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Arithmetic operators

#include <cstddef>
#include <iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
TEST_CONSTEXPR_CXX14 bool tests() {
  int array[] = {40, 41, 42, 43, 44};
  int* b      = array + 0;
  int* e      = array + 5;

  // ++it
  {
    std::__bounded_iter<Iter> iter    = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
    std::__bounded_iter<Iter>& result = ++iter;
    assert(&result == &iter);
    assert(*iter == 41);
  }
  // it++
  {
    std::__bounded_iter<Iter> iter   = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
    std::__bounded_iter<Iter> result = iter++;
    assert(*result == 40);
    assert(*iter == 41);
  }
  // --it
  {
    std::__bounded_iter<Iter> iter    = std::__make_bounded_iter(Iter(b + 3), Iter(b), Iter(e));
    std::__bounded_iter<Iter>& result = --iter;
    assert(&result == &iter);
    assert(*iter == 42);
  }
  // it--
  {
    std::__bounded_iter<Iter> iter   = std::__make_bounded_iter(Iter(b + 3), Iter(b), Iter(e));
    std::__bounded_iter<Iter> result = iter--;
    assert(*result == 43);
    assert(*iter == 42);
  }
  // it += n
  {
    std::__bounded_iter<Iter> iter    = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
    std::__bounded_iter<Iter>& result = (iter += 3);
    assert(&result == &iter);
    assert(*iter == 43);
  }
  // it + n
  {
    std::__bounded_iter<Iter> iter   = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
    std::__bounded_iter<Iter> result = iter + 3;
    assert(*iter == 40);
    assert(*result == 43);
  }
  // n + it
  {
    std::__bounded_iter<Iter> iter   = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
    std::__bounded_iter<Iter> result = 3 + iter;
    assert(*iter == 40);
    assert(*result == 43);
  }
  // it -= n
  {
    std::__bounded_iter<Iter> iter    = std::__make_bounded_iter(Iter(b + 3), Iter(b), Iter(e));
    std::__bounded_iter<Iter>& result = (iter -= 3);
    assert(&result == &iter);
    assert(*iter == 40);
  }
  // it - n
  {
    std::__bounded_iter<Iter> iter   = std::__make_bounded_iter(Iter(b + 3), Iter(b), Iter(e));
    std::__bounded_iter<Iter> result = iter - 3;
    assert(*iter == 43);
    assert(*result == 40);
  }
  // it - it
  {
    std::__bounded_iter<Iter> iter1 = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
    std::__bounded_iter<Iter> iter2 = std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));
    std::ptrdiff_t result           = iter2 - iter1;
    assert(result == 5);
  }

  return true;
}

int main(int, char**) {
  tests<int*>();
#if TEST_STD_VER > 11
  static_assert(tests<int*>(), "");
#endif

#if TEST_STD_VER > 17
  tests<contiguous_iterator<int*> >();
  static_assert(tests<contiguous_iterator<int*> >(), "");
#endif

  return 0;
}
