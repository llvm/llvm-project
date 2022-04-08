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
#include <numeric>
#include <random>
#include <cassert>
#include <vector>
#include <deque>

#include "test_macros.h"

std::mt19937 randomness;

template <class Container, class RI>
void
test_sort_helper(RI f, RI l)
{
    if (f != l)
    {
        Container save(l - f);
        do
        {
            std::copy(f, l, save.begin());
            std::sort(save.begin(), save.end());
            assert(std::is_sorted(save.begin(), save.end()));
            assert(std::is_permutation(save.begin(), save.end(), f));
        } while (std::next_permutation(f, l));
    }
}

template <class T>
void set_value(T& dest, int value)
{
    dest = value;
}

inline void set_value(std::pair<int, int>& dest, int value)
{
    dest.first = value;
    dest.second = value;
}

template <class Container, class RI>
void
test_sort_driver_driver(RI f, RI l, int start, RI real_last)
{
    for (RI i = l; i > f + start;)
    {
        set_value(*--i, start);
        if (f == i)
        {
            test_sort_helper<Container>(f, real_last);
        }
        if (start > 0)
            test_sort_driver_driver<Container>(f, i, start-1, real_last);
    }
}

template <class Container, class RI>
void
test_sort_driver(RI f, RI l, int start)
{
    test_sort_driver_driver<Container>(f, l, start, l);
}

template <class Container, int sa>
void
test_sort_()
{
    Container ia(sa);
    for (int i = 0; i < sa; ++i)
    {
        test_sort_driver<Container>(ia.begin(), ia.end(), i);
    }
}

template <class T>
T increment_or_reset(T value, int max_value)
{
    return value == max_value - 1 ? 0 : value + 1;
}

inline std::pair<int, int> increment_or_reset(std::pair<int, int> value,
                                              int max_value)
{
    int new_value = value.first + 1;
    if (new_value == max_value)
    {
        new_value = 0;
    }
    return std::make_pair(new_value, new_value);
}

template <class Container>
void
test_larger_sorts(int N, int M)
{
    using Iter = typename Container::iterator;
    using ValueType = typename Container::value_type;
    assert(N != 0);
    assert(M != 0);
    // create container of length N filled with M different objects
    Container array(N);
    ValueType x = ValueType();
    for (int i = 0; i < N; ++i)
    {
        array[i] = x;
        x = increment_or_reset(x, M);
    }
    Container original = array;
    Iter iter = array.begin();
    Iter original_iter = original.begin();

    // test saw tooth pattern
    std::sort(iter, iter+N);
    assert(std::is_sorted(iter, iter+N));
    assert(std::is_permutation(iter, iter+N, original_iter));
    // test random pattern
    std::shuffle(iter, iter+N, randomness);
    std::sort(iter, iter+N);
    assert(std::is_sorted(iter, iter+N));
    assert(std::is_permutation(iter, iter+N, original_iter));
    // test sorted pattern
    std::sort(iter, iter+N);
    assert(std::is_sorted(iter, iter+N));
    assert(std::is_permutation(iter, iter+N, original_iter));
    // test reverse sorted pattern
    std::reverse(iter, iter+N);
    std::sort(iter, iter+N);
    assert(std::is_sorted(iter, iter+N));
    assert(std::is_permutation(iter, iter+N, original_iter));
    // test swap ranges 2 pattern
    std::swap_ranges(iter, iter+N/2, iter+N/2);
    std::sort(iter, iter+N);
    assert(std::is_sorted(iter, iter+N));
    assert(std::is_permutation(iter, iter+N, original_iter));
    // test reverse swap ranges 2 pattern
    std::reverse(iter, iter+N);
    std::swap_ranges(iter, iter+N/2, iter+N/2);
    std::sort(iter, iter+N);
    assert(std::is_sorted(iter, iter+N));
    assert(std::is_permutation(iter, iter+N, original_iter));
}

template <class Container>
void
test_larger_sorts(int N)
{
    test_larger_sorts<Container>(N, 1);
    test_larger_sorts<Container>(N, 2);
    test_larger_sorts<Container>(N, 3);
    test_larger_sorts<Container>(N, N/2-1);
    test_larger_sorts<Container>(N, N/2);
    test_larger_sorts<Container>(N, N/2+1);
    test_larger_sorts<Container>(N, N-2);
    test_larger_sorts<Container>(N, N-1);
    test_larger_sorts<Container>(N, N);
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

template <class Container>
void run_sort_tests()
{
    // test null range
    using ValueType = typename Container::value_type;
    ValueType d = ValueType();
    std::sort(&d, &d);

    // exhaustively test all possibilities up to length 8
    test_sort_<Container, 1>();
    test_sort_<Container, 2>();
    test_sort_<Container, 3>();
    test_sort_<Container, 4>();
    test_sort_<Container, 5>();
    test_sort_<Container, 6>();
    test_sort_<Container, 7>();
    test_sort_<Container, 8>();

    test_larger_sorts<Container>(256);
    test_larger_sorts<Container>(257);
    test_larger_sorts<Container>(499);
    test_larger_sorts<Container>(500);
    test_larger_sorts<Container>(997);
    test_larger_sorts<Container>(1000);
    test_larger_sorts<Container>(1009);
}

int main(int, char**)
{
    // test various combinations of contiguous/non-contiguous containers with
    // arithmetic/non-arithmetic types
    run_sort_tests<std::vector<int> >();
    run_sort_tests<std::deque<int> >();
    run_sort_tests<std::vector<std::pair<int, int> > >();

    test_pointer_sort();
    test_adversarial_quicksort(1 << 20);

    return 0;
}
