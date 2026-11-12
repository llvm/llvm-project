//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=3000000

// void splice_after(const_iterator p, forward_list&& x,
//                   const_iterator first, const_iterator last); // constexpr since C++26

#include <stddef.h>
#include <forward_list>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "min_allocator.h"

typedef std::ptrdiff_t T;
TEST_CONSTEXPR const T t1[]                 = {0, 1, 2, 3, 4, 5, 6, 7};
TEST_CONSTEXPR const T t2[]                 = {10, 11, 12, 13, 14, 15};
TEST_CONSTEXPR const std::ptrdiff_t size_t1 = std::end(t1) - std::begin(t1);
TEST_CONSTEXPR const std::ptrdiff_t size_t2 = std::end(t2) - std::begin(t2);

template <class C>
TEST_CONSTEXPR_CXX26 void testd(const C& c, std::ptrdiff_t p, ptrdiff_t f, ptrdiff_t l) {
  typename C::const_iterator i = c.begin();
  std::ptrdiff_t n1            = 0;
  for (; n1 < p; ++n1, ++i)
    assert(*i == t1[n1]);
  for (std::ptrdiff_t n2 = f; n2 < l - 1; ++n2, ++i)
    assert(*i == t2[n2]);
  for (; n1 < size_t1; ++n1, ++i)
    assert(*i == t1[n1]);
  assert(std::distance(c.begin(), c.end()) == size_t1 + (l > f + 1 ? l - 1 - f : 0));
}

template <class C>
TEST_CONSTEXPR_CXX26 void tests(const C& c, std::ptrdiff_t p, ptrdiff_t f, ptrdiff_t l) {
  typename C::const_iterator i = c.begin();
  std::ptrdiff_t n             = 0;
  std::ptrdiff_t d             = l > f + 1 ? l - 1 - f : 0;
  if (d == 0 || p == f) {
    for (n = 0; n < size_t1; ++n, ++i)
      assert(*i == t1[n]);
  } else if (p < f) {
    for (n = 0; n < p; ++n, ++i)
      assert(*i == t1[n]);
    for (n = f; n < l - 1; ++n, ++i)
      assert(*i == t1[n]);
    for (n = p; n < f; ++n, ++i)
      assert(*i == t1[n]);
    for (n = l - 1; n < size_t1; ++n, ++i)
      assert(*i == t1[n]);
  } else // p > f
  {
    for (n = 0; n < f; ++n, ++i)
      assert(*i == t1[n]);
    for (n = l - 1; n < p; ++n, ++i)
      assert(*i == t1[n]);
    for (n = f; n < l - 1; ++n, ++i)
      assert(*i == t1[n]);
    for (n = p; n < size_t1; ++n, ++i)
      assert(*i == t1[n]);
  }
  assert(std::distance(c.begin(), c.end()) == size_t1);
}

TEST_CONSTEXPR_CXX26 bool test() {
  {
    // splicing different containers
    typedef std::forward_list<T> C;
    for (std::ptrdiff_t f = 0; f <= size_t2 + 1; ++f) {
      for (std::ptrdiff_t l = f; l <= size_t2 + 1; ++l) {
        for (std::ptrdiff_t p = 0; p <= size_t1; ++p) {
          C c1(std::begin(t1), std::end(t1));
          C c2(std::begin(t2), std::end(t2));

          c1.splice_after(std::next(c1.cbefore_begin(), p),
                          std::move(c2),
                          std::next(c2.cbefore_begin(), f),
                          std::next(c2.cbefore_begin(), l));
          testd(c1, p, f, l);
        }
      }
    }

    // splicing within same container
    for (std::ptrdiff_t f = 0; f <= size_t1 + 1; ++f) {
      for (std::ptrdiff_t l = f; l <= size_t1; ++l) {
        for (std::ptrdiff_t p = 0; p <= f; ++p) {
          C c1(std::begin(t1), std::end(t1));

          c1.splice_after(std::next(c1.cbefore_begin(), p),
                          std::move(c1),
                          std::next(c1.cbefore_begin(), f),
                          std::next(c1.cbefore_begin(), l));
          tests(c1, p, f, l);
        }
        for (std::ptrdiff_t p = l; p <= size_t1; ++p) {
          C c1(std::begin(t1), std::end(t1));

          c1.splice_after(std::next(c1.cbefore_begin(), p),
                          std::move(c1),
                          std::next(c1.cbefore_begin(), f),
                          std::next(c1.cbefore_begin(), l));
          tests(c1, p, f, l);
        }
      }
    }
  }
#if TEST_STD_VER >= 11
  {
    // splicing different containers
    typedef std::forward_list<T, min_allocator<T>> C;
    for (std::ptrdiff_t f = 0; f <= size_t2 + 1; ++f) {
      for (std::ptrdiff_t l = f; l <= size_t2 + 1; ++l) {
        for (std::ptrdiff_t p = 0; p <= size_t1; ++p) {
          C c1(std::begin(t1), std::end(t1));
          C c2(std::begin(t2), std::end(t2));

          c1.splice_after(std::next(c1.cbefore_begin(), p),
                          std::move(c2),
                          std::next(c2.cbefore_begin(), f),
                          std::next(c2.cbefore_begin(), l));
          testd(c1, p, f, l);
        }
      }
    }

    // splicing within same container
    for (std::ptrdiff_t f = 0; f <= size_t1 + 1; ++f) {
      for (std::ptrdiff_t l = f; l <= size_t1; ++l) {
        for (std::ptrdiff_t p = 0; p <= f; ++p) {
          C c1(std::begin(t1), std::end(t1));

          c1.splice_after(std::next(c1.cbefore_begin(), p),
                          std::move(c1),
                          std::next(c1.cbefore_begin(), f),
                          std::next(c1.cbefore_begin(), l));
          tests(c1, p, f, l);
        }
        for (std::ptrdiff_t p = l; p <= size_t1; ++p) {
          C c1(std::begin(t1), std::end(t1));

          c1.splice_after(std::next(c1.cbefore_begin(), p),
                          std::move(c1),
                          std::next(c1.cbefore_begin(), f),
                          std::next(c1.cbefore_begin(), l));
          tests(c1, p, f, l);
        }
      }
    }
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
