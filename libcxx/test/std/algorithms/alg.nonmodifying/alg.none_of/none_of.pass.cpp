//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class InputIterator, class Predicate>
// constexpr bool none_of(InputIterator first, InputIterator last, Predicate pred); // constexpr since C++20

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

struct is_odd {
  TEST_CONSTEXPR bool operator()(const int& i) const { return i % 2 != 0; }
};

struct counting_predicate {
  int* times_applied;
  bool result;
  TEST_CONSTEXPR counting_predicate(int* counter, bool r) : times_applied(counter), result(r) {}
  TEST_CONSTEXPR_CXX14 bool operator()(int) const {
    ++*times_applied;
    return result;
  }
};

struct Test {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    { // an empty range vacuously satisfies none_of
      int a[] = {1, 3, 5, 7};
      // The return type is always `bool`, regardless of the iterator category.
      ASSERT_SAME_TYPE(bool, decltype(std::none_of(Iter(a), Iter(a), is_odd())));
      assert(std::none_of(Iter(a), Iter(a), is_odd()));
    }

    { // no element satisfies the predicate
      int a[] = {2, 4, 6, 8};
      assert(std::none_of(Iter(a), Iter(a + 4), is_odd()));
    }

    { // every element satisfies the predicate
      int a[] = {1, 3, 5, 7};
      assert(!std::none_of(Iter(a), Iter(a + 4), is_odd()));
    }

    { // the only satisfying element is the first one
      int a[] = {1, 2, 4, 6};
      assert(!std::none_of(Iter(a), Iter(a + 4), is_odd()));
    }

    { // the only satisfying element is in the middle
      int a[] = {2, 4, 5, 8};
      assert(!std::none_of(Iter(a), Iter(a + 4), is_odd()));
    }

    { // the only satisfying element is the last one
      int a[] = {2, 4, 6, 7};
      assert(!std::none_of(Iter(a), Iter(a + 4), is_odd()));
    }

    { // a single-element range, satisfying and not satisfying the predicate
      int satisfies     = 1;
      int not_satisfies = 2;
      assert(!std::none_of(Iter(&satisfies), Iter(&satisfies + 1), is_odd()));
      assert(std::none_of(Iter(&not_satisfies), Iter(&not_satisfies + 1), is_odd()));
    }
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<int*>(), Test());

  { // the predicate is applied at most `last - first` times (complexity requirement)
    int a[]     = {1, 2, 3, 4, 5};
    int applied = 0;
    assert(std::none_of(a, a + 5, counting_predicate(&applied, false)));
    assert(applied == 5);
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
