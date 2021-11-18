//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter>
//         && LessThanComparable<Iter::value_type>
//   void
//   sort(Iter first, Iter last);

#include <algorithm>
#include <iterator>
#include <random>
#include <cassert>

#include "test_macros.h"

std::mt19937 randomness;

template <class RI>
void
test_sort_helper(RI f, RI l)
{
    typedef typename std::iterator_traits<RI>::value_type value_type;
    typedef typename std::iterator_traits<RI>::difference_type difference_type;

    if (f != l)
    {
        difference_type len = l - f;
        value_type* save(new value_type[len]);
        do
        {
            std::copy(f, l, save);
            std::sort(save, save+len);
            assert(std::is_sorted(save, save+len));
        } while (std::next_permutation(f, l));
        delete [] save;
    }
}

template <class RI>
void
test_sort_driver_driver(RI f, RI l, int start, RI real_last)
{
    for (RI i = l; i > f + start;)
    {
        *--i = start;
        if (f == i)
        {
            test_sort_helper(f, real_last);
        }
    if (start > 0)
        test_sort_driver_driver(f, i, start-1, real_last);
    }
}

template <class RI>
void
test_sort_driver(RI f, RI l, int start)
{
    test_sort_driver_driver(f, l, start, l);
}

template <int sa>
void
test_sort_()
{
    int ia[sa];
    for (int i = 0; i < sa; ++i)
    {
        test_sort_driver(ia, ia+sa, i);
    }
}

void
test_larger_sorts(int N, int M)
{
    assert(N != 0);
    assert(M != 0);
    // create array length N filled with M different numbers
    int* array = new int[N];
    int x = 0;
    for (int i = 0; i < N; ++i)
    {
        array[i] = x;
        if (++x == M)
            x = 0;
    }
    // test saw tooth pattern
    std::sort(array, array+N);
    assert(std::is_sorted(array, array+N));
    // test random pattern
    std::shuffle(array, array+N, randomness);
    std::sort(array, array+N);
    assert(std::is_sorted(array, array+N));
    // test sorted pattern
    std::sort(array, array+N);
    assert(std::is_sorted(array, array+N));
    // test reverse sorted pattern
    std::reverse(array, array+N);
    std::sort(array, array+N);
    assert(std::is_sorted(array, array+N));
    // test swap ranges 2 pattern
    std::swap_ranges(array, array+N/2, array+N/2);
    std::sort(array, array+N);
    assert(std::is_sorted(array, array+N));
    // test reverse swap ranges 2 pattern
    std::reverse(array, array+N);
    std::swap_ranges(array, array+N/2, array+N/2);
    std::sort(array, array+N);
    assert(std::is_sorted(array, array+N));
    delete [] array;
}

void
test_larger_sorts(int N)
{
    test_larger_sorts(N, 1);
    test_larger_sorts(N, 2);
    test_larger_sorts(N, 3);
    test_larger_sorts(N, N/2-1);
    test_larger_sorts(N, N/2);
    test_larger_sorts(N, N/2+1);
    test_larger_sorts(N, N-2);
    test_larger_sorts(N, N-1);
    test_larger_sorts(N, N);
}

void
test_pointer_sort()
{
    static const int array_size = 10;
    const int v[array_size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int *pv[array_size];
    for (int i = 0; i < array_size; i++) {
      pv[i] = &v[array_size - 1 - i];
    }
    std::sort(pv, pv + array_size);
    assert(*pv[0] == v[0]);
    assert(*pv[1] == v[1]);
    assert(*pv[array_size - 1] == v[array_size - 1]);
}

// test_adversarial_quicksort generates a vector with values arranged in such a
// way that they would invoke O(N^2) behavior on any quick sort implementation
// that satisifies certain conditions.  Details are available in the following
// paper:
// "A Killer Adversary for Quicksort", M. D. McIlroy, Software—Practice &
// ExperienceVolume 29 Issue 4 April 10, 1999 pp 341–344.
// https://dl.acm.org/doi/10.5555/311868.311871.
struct AdversaryComparator {
  AdversaryComparator(int N, std::vector<int>& input) : gas(N - 1), V(input) {
    V.resize(N);
    // Populate all positions in the generated input to gas to indicate that
    // none of the values have been fixed yet.
    for (int i = 0; i < N; ++i)
      V[i] = gas;
  }

  bool operator()(int x, int y) {
    if (V[x] == gas && V[y] == gas) {
      // We are comparing two inputs whose value is still to be decided.
      if (x == candidate) {
        V[x] = nsolid++;
      } else {
        V[y] = nsolid++;
      }
    }
    if (V[x] == gas) {
      candidate = x;
    } else if (V[y] == gas) {
      candidate = y;
    }
    return V[x] < V[y];
  }

private:
  // If an element is equal to gas, it indicates that the value of the element
  // is still to be decided and may change over the course of time.
  const int gas;
  // This is a reference so that we can manipulate the input vector later.
  std::vector<int>& V;
  // Candidate for the pivot position.
  int candidate = 0;
  int nsolid = 0;
};

void test_adversarial_quicksort(int N) {
  assert(N > 0);
  std::vector<int> ascVals(N);
  // Fill up with ascending values from 0 to N-1.  These will act as indices
  // into V.
  std::iota(ascVals.begin(), ascVals.end(), 0);
  std::vector<int> V;
  AdversaryComparator comp(N, V);
  std::sort(ascVals.begin(), ascVals.end(), comp);
  std::sort(V.begin(), V.end());
  assert(std::is_sorted(V.begin(), V.end()));
}

int main(int, char**)
{
    // test null range
    int d = 0;
    std::sort(&d, &d);
    // exhaustively test all possibilities up to length 8
    test_sort_<1>();
    test_sort_<2>();
    test_sort_<3>();
    test_sort_<4>();
    test_sort_<5>();
    test_sort_<6>();
    test_sort_<7>();
    test_sort_<8>();

    test_larger_sorts(256);
    test_larger_sorts(257);
    test_larger_sorts(499);
    test_larger_sorts(500);
    test_larger_sorts(997);
    test_larger_sorts(1000);
    test_larger_sorts(1009);

    test_pointer_sort();
    test_adversarial_quicksort(1 << 20);

    return 0;
}
