//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter1, ForwardIterator Iter2>
//   requires HasSwap<Iter1::reference, Iter2::reference>
//   Iter2
//   swap_ranges(Iter1 first1, Iter1 last1, Iter2 first2);

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

struct TestPtr {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(types::forward_iterator_list<int*>(), TestImpl<Iter>());
  }

  template <class Iter1>
  struct TestImpl {
    template <class Iter2>
    TEST_CONSTEXPR_CXX20 void operator()() {
      { // Basic test case: swapping three elements between two arrays
        int a[] = {1, 2, 3};
        int b[] = {4, 5, 6};
        Iter2 r = std::swap_ranges(Iter1(a), Iter1(a + 3), Iter2(b));
        assert(base(r) == b + 3);
        assert(a[0] == 4 && a[1] == 5 && a[2] == 6);
        assert(b[0] == 1 && b[1] == 2 && b[2] == 3);
      }
      { // Large-scale test: swapping 100 elements between two different containers
        const int N = 100;
        std::array<int, N> a;
        std::vector<int> b(N + 2, 42);
        b.front() = 1;
        b.back()  = -1;
        for (int i = 0; i < N; ++i)
          a[i] = i * i + 1;
        Iter2 r = std::swap_ranges(Iter1(a.data()), Iter1(a.data() + N), Iter2(b.data() + 1));
        assert(base(r) == b.data() + N + 1);
        assert(b.front() == 1); // Ensure that the unswapped portion remains unchanged
        assert(b.back() == -1);
        for (int i = 0; i < N; ++i) {
          assert(a[i] == 42);
          assert(b[i + 1] == i * i + 1);
        }
      }
    }
  };
};

#if TEST_STD_VER >= 11
struct TestUniquePtr {
  template <class Iter>
  TEST_CONSTEXPR_CXX23 void operator()() {
    types::for_each(types::forward_iterator_list<std::unique_ptr<int>*>(), TestImpl<Iter>());
  }

  template <class Iter1>
  struct TestImpl {
    template <class Iter2>
    TEST_CONSTEXPR_CXX23 void operator()() {
      std::unique_ptr<int> a[3];
      for (int k = 0; k < 3; ++k)
        a[k].reset(new int(k + 1));
      std::unique_ptr<int> b[3];
      for (int k = 0; k < 3; ++k)
        b[k].reset(new int(k + 4));
      Iter2 r = std::swap_ranges(Iter1(a), Iter1(a + 3), Iter2(b));
      assert(base(r) == b + 3);
      assert(*a[0] == 4 && *a[1] == 5 && *a[2] == 6);
      assert(*b[0] == 1 && *b[1] == 2 && *b[2] == 3);
    }
  };
};
#endif

template <template <class> class Iter1, template <class> class Iter2>
TEST_CONSTEXPR_CXX20 bool test_simple_cases() {
  {
    int a[2] = {1, 2};
    int b[2] = {4, 5};
    std::swap_ranges(Iter1<int*>(a), Iter1<int*>(a + 2), Iter2<int*>(b));
    assert(a[0] == 4 && a[1] == 5);
    assert(b[0] == 1 && b[1] == 2);
  }
  {
    std::array<int, 3> a = {1, 2, 3}, a0 = a;
    std::array<int, 3> b = {4, 5, 6}, b0 = b;
    using It1 = Iter1<std::array<int, 3>::iterator>;
    using It2 = Iter2<std::array<int, 3>::iterator>;
    std::swap_ranges(It1(a.begin()), It1(a.end()), It2(b.begin()));
    assert(a == b0);
    assert(b == a0);
  }
  {
    std::array<std::array<int, 2>, 2> a = {{{0, 1}, {2, 3}}}, a0 = a;
    std::array<std::array<int, 2>, 2> b = {{{9, 8}, {7, 6}}}, b0 = b;
    using It1 = Iter1<std::array<std::array<int, 2>, 2>::iterator>;
    using It2 = Iter2<std::array<std::array<int, 2>, 2>::iterator>;
    std::swap_ranges(It1(a.begin()), It1(a.end()), It2(b.begin()));
    assert(a == b0);
    assert(b == a0);
  }
  {
    std::array<std::array<int, 3>, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}}, a0 = a;
    std::array<std::array<int, 3>, 3> b = {{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}}, b0 = b;
    using It1 = Iter1<std::array<std::array<int, 3>, 3>::iterator>;
    using It2 = Iter2<std::array<std::array<int, 3>, 3>::iterator>;
    std::swap_ranges(It1(a.begin()), It1(a.end()), It2(b.begin()));
    assert(a == b0);
    assert(b == a0);
  }

  return true;
}

template <std::size_t N>
TEST_CONSTEXPR_CXX20 void test_vector_bool() {
  std::vector<bool> f(N, false), t(N, true);
  { // Test swap_ranges() with aligned bytes
    std::vector<bool> f1 = f, t1 = t;
    std::swap_ranges(f1.begin(), f1.end(), t1.begin());
    assert(f1 == t);
    assert(t1 == f);
  }
  { // Test swap_ranges() with unaligned bytes
    std::vector<bool> f1(N, false), t1(N + 8, true);
    std::swap_ranges(f1.begin(), f1.end(), t1.begin() + 4);
    assert(std::equal(f1.begin(), f1.end(), t.begin()));
    assert(std::equal(t1.begin() + 4, t1.end() - 4, f.begin()));
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_simple_cases<forward_iterator, forward_iterator>();
  test_simple_cases<forward_iterator, bidirectional_iterator>();
  test_simple_cases<forward_iterator, random_access_iterator>();
  test_simple_cases<bidirectional_iterator, forward_iterator>();
  test_simple_cases<bidirectional_iterator, bidirectional_iterator>();
  test_simple_cases<bidirectional_iterator, random_access_iterator>();
  test_simple_cases<random_access_iterator, random_access_iterator>();
#if TEST_STD_VER >= 20
  test_simple_cases<std::type_identity_t, std::type_identity_t>();
#endif

  types::for_each(types::forward_iterator_list<int*>(), TestPtr());

#if TEST_STD_VER >= 11
  // We can't test unique_ptr in constant evaluation before C++23 as it's constexpr only since C++23.
  if (TEST_STD_AT_LEAST_23_OR_RUNTIME_EVALUATED)
    types::for_each(types::forward_iterator_list<std::unique_ptr<int>*>(), TestUniquePtr());
#endif

  { // Test vector<bool>::iterator optimization
    test_vector_bool<8>();
    test_vector_bool<19>();
    test_vector_bool<32>();
    test_vector_bool<49>();
    test_vector_bool<64>();
    test_vector_bool<199>();
    test_vector_bool<256>();
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
