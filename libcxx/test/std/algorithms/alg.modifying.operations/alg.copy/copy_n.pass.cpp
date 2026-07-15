//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter   // constexpr after C++17
//   copy_n(InIter first, InIter::difference_type n, OutIter result);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"
#include "user_defined_integral.h"

typedef UserDefinedIntegral<unsigned> UDI;

// A minimal single-pass input iterator that counts how many times it is advanced. Used to verify that
// copy_n reads exactly n elements (advancing the iterator only n - 1 times).
struct CountingInputIterator {
  using iterator_category = std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = long;
  using pointer           = const int*;
  using reference         = const int&;

  const int* ptr_;
  int* increments_;

  TEST_CONSTEXPR_CXX14 CountingInputIterator(const int* ptr, int* increment_counter)
      : ptr_(ptr), increments_(increment_counter) {}

  TEST_CONSTEXPR_CXX14 reference operator*() const { return *ptr_; }
  TEST_CONSTEXPR_CXX14 CountingInputIterator& operator++() {
    ++ptr_;
    ++*increments_;
    return *this;
  }
  TEST_CONSTEXPR_CXX14 CountingInputIterator operator++(int) {
    CountingInputIterator __tmp = *this;
    ++*this;
    return __tmp;
  }
};

class PaddedBase {
public:
  TEST_CONSTEXPR PaddedBase(std::int16_t a, std::int8_t b) : a_(a), b_(b) {}

  std::int16_t a_;
  std::int8_t b_;
};

class Derived : public PaddedBase {
public:
  TEST_CONSTEXPR Derived(std::int16_t a, std::int8_t b, std::int8_t c) : PaddedBase(a, b), c_(c) {}

  std::int8_t c_;
};

struct TestIterators {
  template <class InIter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(
        types::concatenate_t<types::cpp17_input_iterator_list<int*>, types::type_list<cpp17_output_iterator<int*> > >(),
        TestImpl<InIter>());
  }

  template <class InIter>
  struct TestImpl {
    template <class OutIter>
    TEST_CONSTEXPR_CXX20 void operator()() {
      const unsigned N = 1000;
      int ia[N]        = {};
      for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
      int ib[N] = {0};

      OutIter r = std::copy_n(InIter(ia), UDI(N / 2), OutIter(ib));
      assert(base(r) == ib + N / 2);
      for (unsigned i = 0; i < N / 2; ++i)
        assert(ia[i] == ib[i]);

      { // A negative count is a no-op that returns the unchanged output iterator.
        // Regression test for https://llvm.org/PR193613.
        int source[] = {1, 2, 3};
        int dest[]   = {-1, -2, -3};
        OutIter ret  = std::copy_n(InIter(source), -5, OutIter(dest));
        assert(base(ret) == dest);
        assert(dest[0] == -1 && dest[1] == -2 && dest[2] == -3);
      }
    }
  };
};

TEST_CONSTEXPR_CXX20 bool test_vector_bool(std::size_t N) {
  std::vector<bool> in(N, false);
  for (std::size_t i = 0; i < N; i += 2)
    in[i] = true;

  { // Test copy with aligned bytes
    std::vector<bool> out(N);
    std::copy_n(in.begin(), N, out.begin());
    assert(in == out);
  }
  { // Test copy with unaligned bytes
    std::vector<bool> out(N + 8);
    std::copy_n(in.begin(), N, out.begin() + 4);
    for (std::size_t i = 0; i < N; ++i)
      assert(out[i + 4] == in[i]);
  }
  { // Negative count test
    std::vector<bool> source(N, true);
    std::vector<bool> dest(N, false);
    std::vector<bool>::iterator r = std::copy_n(source.begin(), -5, dest.begin());
    assert(r == dest.begin());
    assert(dest == std::vector<bool>(N, false));
  }

  return true;
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<const int*>(), TestIterators());

  { // Make sure that padding bits aren't copied
    Derived src(1, 2, 3);
    Derived dst(4, 5, 6);
    std::copy_n(static_cast<PaddedBase*>(&src), 1, static_cast<PaddedBase*>(&dst));
    assert(dst.a_ == 1);
    assert(dst.b_ == 2);
    assert(dst.c_ == 6);
  }

  { // Make sure that overlapping ranges can be copied
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::copy_n(a + 3, 7, a);
    int expected[] = {4, 5, 6, 7, 8, 9, 10, 8, 9, 10};
    assert(std::equal(a, a + 10, expected));
  }

  { // Test vector<bool>::iterator optimization
    assert(test_vector_bool(8));
    assert(test_vector_bool(19));
    assert(test_vector_bool(32));
    assert(test_vector_bool(49));
    assert(test_vector_bool(64));
    assert(test_vector_bool(199));
    assert(test_vector_bool(256));
  }

  { // For a single-pass input iterator, copy_n reads exactly n elements -- advancing the iterator only
    // n - 1 times. For n == 0, it shouldn't advance the iterator. See 99847d2bf132.
    int in[] = {1, 2, 3, 4, 5};

    { // n == 0 is a no-op
      int out[3]     = {-1, -2, -3};
      int increments = 0;
      int* r         = std::copy_n(CountingInputIterator(in, &increments), 0, out);
      assert(r == out);
      assert(increments == 0);
      assert(out[0] == -1 && out[1] == -2 && out[2] == -3);
    }
    { // n > 0 advances the iterator exactly n - 1 times
      int out[3]     = {0, 0, 0};
      int increments = 0;
      int* r         = std::copy_n(CountingInputIterator(in, &increments), 3, out);
      assert(r == out + 3);
      assert(increments == 2);
      assert(out[0] == 1 && out[1] == 2 && out[2] == 3);
    }
  }

  return true;
}

int main(int, char**) {
  test();

#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
