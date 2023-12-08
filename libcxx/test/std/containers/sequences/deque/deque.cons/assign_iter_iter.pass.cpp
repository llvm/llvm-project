//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// template <class InputIterator>
//   void assign(InputIterator f, InputIterator l);

#include "asan_testing.h"
#include <deque>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"
#if TEST_STD_VER >= 11
#include "emplace_constructible.h"
#endif

template <class C>
C
make(int size, int start = 0 )
{
    const int b = 4096 / sizeof(int);
    int init = 0;
    if (start > 0)
    {
        init = (start+1) / b + ((start+1) % b != 0);
        init *= b;
        --init;
    }
    C c(init, 0);
    for (int i = 0; i < init-start; ++i)
        c.pop_back();
    for (int i = 0; i < size; ++i)
        c.push_back(i);
    for (int i = 0; i < start; ++i)
        c.pop_front();
    return c;
}

template <class C>
void
test(C& c1, const C& c2)
{
    c1.assign(c2.begin(), c2.end());
    assert(static_cast<std::size_t>(std::distance(c1.begin(), c1.end())) == c1.size());
    assert(c1 == c2);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c1));
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c2));
}

template <class C>
void
testN(int start, int N, int M)
{
    C c1 = make<C>(N, start);
    C c2 = make<C>(M);
    test(c1, c2);
}

template <class C>
void
testI(C& c1, const C& c2)
{
    typedef typename C::const_iterator CI;
    typedef cpp17_input_iterator<CI> ICI;
    c1.assign(ICI(c2.begin()), ICI(c2.end()));
    assert(static_cast<std::size_t>(std::distance(c1.begin(), c1.end())) == c1.size());
    assert(c1 == c2);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c1));
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c2));
}

template <class C>
void
testNI(int start, int N, int M)
{
    C c1 = make<C>(N, start);
    C c2 = make<C>(M);
    testI(c1, c2);
}

void basic_test()
{
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                testN<std::deque<int> >(rng[i], rng[j], rng[k]);
    testNI<std::deque<int> >(1500, 2000, 1000);
    }
#if TEST_STD_VER >= 11
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                testN<std::deque<int, min_allocator<int>> >(rng[i], rng[j], rng[k]);
    testNI<std::deque<int, min_allocator<int>> >(1500, 2000, 1000);
    }
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                testN<std::deque<int, safe_allocator<int>> >(rng[i], rng[j], rng[k]);
    testNI<std::deque<int, safe_allocator<int>> >(1500, 2000, 1000);
    }
#endif
}

template <class It>
void test_emplacable_concept() {
#if TEST_STD_VER >= 11
  int arr1[] = {42};
  int arr2[] = {1, 101, 42};
  {
    using T = EmplaceConstructibleMoveableAndAssignable<int>;
    {
      std::deque<T> v;
      v.assign(It(arr1), It(std::end(arr1)));
      assert(v[0].value == 42);
    }
    {
      std::deque<T> v;
      v.assign(It(arr2), It(std::end(arr2)));
      assert(v[0].value == 1);
      assert(v[1].value == 101);
      assert(v[2].value == 42);
    }
  }
#endif
}

void test_iterators() {
  test_emplacable_concept<cpp17_input_iterator<int*> >();
  test_emplacable_concept<forward_iterator<int*> >();
  test_emplacable_concept<bidirectional_iterator<int*> >();
  test_emplacable_concept<random_access_iterator<int*> >();
#if TEST_STD_VER > 17
  test_emplacable_concept<contiguous_iterator<int*> >();
#endif
  test_emplacable_concept<int*>();
}

int main(int, char**) {
  basic_test();

  return 0;
}
