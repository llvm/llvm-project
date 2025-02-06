//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03 && !stdlib=libc++

// <algorithm>

// template<BidirectionalIterator InIter, BidirectionalIterator OutIter>
//   requires OutputIterator<OutIter, RvalueOf<InIter::reference>::type>
//   OutIter
//   move_backward(InIter first, InIter last, OutIter result);

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <vector>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

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

template <class InIter>
struct Test {
  template <class OutIter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    const unsigned N = 1000;
    int ia[N]        = {};
    for (unsigned i = 0; i < N; ++i)
      ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::move_backward(InIter(ia), InIter(ia + N), OutIter(ib + N));
    assert(base(r) == ib);
    for (unsigned i = 0; i < N; ++i)
      assert(ia[i] == ib[i]);
  }
};

struct TestOutIters {
  template <class InIter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(types::concatenate_t<types::bidirectional_iterator_list<int*> >(), Test<InIter>());
  }
};

template <class InIter>
struct Test1 {
  template <class OutIter>
  TEST_CONSTEXPR_CXX23 void operator()() {
    const unsigned N = 100;
    std::unique_ptr<int> ia[N];
    for (unsigned i = 0; i < N; ++i)
      ia[i].reset(new int(i));
    std::unique_ptr<int> ib[N];

    OutIter r = std::move_backward(InIter(ia), InIter(ia + N), OutIter(ib + N));
    assert(base(r) == ib);
    for (unsigned i = 0; i < N; ++i)
      assert(*ib[i] == static_cast<int>(i));
  }
};

struct Test1OutIters {
  template <class InIter>
  TEST_CONSTEXPR_CXX23 void operator()() {
    types::for_each(
        types::concatenate_t<types::bidirectional_iterator_list<std::unique_ptr<int>*> >(), Test1<InIter>());
  }
};

TEST_CONSTEXPR_CXX20 bool test_vector_bool(std::size_t N) {
  std::vector<bool> v(N, false);
  for (std::size_t i = 0; i < N; i += 2)
    v[i] = true;

  { // Test move_backward with aligned bytes
    std::vector<bool> in(v);
    std::vector<bool> out(N);
    std::move_backward(in.begin(), in.end(), out.end());
    assert(out == v);
  }
  { // Test move_backward with unaligned bytes
    std::vector<bool> in(v);
    std::vector<bool> out(N);
    std::move_backward(in.begin(), in.end() - 4, out.end());
    for (std::size_t i = 0; i < N - 4; ++i)
      assert(out[i + 4] == v[i]);
  }

  return true;
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::bidirectional_iterator_list<int*>(), TestOutIters());
  if (TEST_STD_AT_LEAST_23_OR_RUNTIME_EVALUATED)
    types::for_each(types::bidirectional_iterator_list<std::unique_ptr<int>*>(), Test1OutIters());

  { // Make sure that padding bits aren't copied
    Derived src(1, 2, 3);
    Derived dst(4, 5, 6);
    std::move_backward(
        static_cast<PaddedBase*>(&src), static_cast<PaddedBase*>(&src) + 1, static_cast<PaddedBase*>(&dst) + 1);
    assert(dst.a_ == 1);
    assert(dst.b_ == 2);
    assert(dst.c_ == 6);
  }

  { // Make sure that overlapping ranges can be copied
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::move_backward(a, a + 7, a + 10);
    int expected[] = {1, 2, 3, 1, 2, 3, 4, 5, 6, 7};
    assert(std::equal(a, a + 10, expected));
  }

  // Make sure that the algorithm works with move-only types
  {
    // When non-trivial
    {
      MoveOnly from[3] = {1, 2, 3};
      MoveOnly to[3]   = {};
      std::move_backward(std::begin(from), std::end(from), std::end(to));
      assert(to[0] == MoveOnly(1));
      assert(to[1] == MoveOnly(2));
      assert(to[2] == MoveOnly(3));
    }
    // When trivial
    {
      TrivialMoveOnly from[3] = {1, 2, 3};
      TrivialMoveOnly to[3]   = {};
      std::move_backward(std::begin(from), std::end(from), std::end(to));
      assert(to[0] == TrivialMoveOnly(1));
      assert(to[1] == TrivialMoveOnly(2));
      assert(to[2] == TrivialMoveOnly(3));
    }
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

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
