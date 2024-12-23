//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<BidirectionalIterator InIter, BidirectionalIterator OutIter>
//   requires OutputIterator<OutIter, InIter::reference>
//   constexpr OutIter   // constexpr after C++17
//   copy_backward(InIter first, InIter last, OutIter result);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"
#include "user_defined_integral.h"

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

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 void test_copy_backward() {
  {
    const unsigned N = 1000;
    int ia[N]        = {};
    for (unsigned i = 0; i < N; ++i)
      ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::copy_backward(InIter(ia), InIter(ia + N), OutIter(ib + N));
    assert(base(r) == ib);
    for (unsigned i = 0; i < N; ++i)
      assert(ia[i] == ib[i]);
  }
}

TEST_CONSTEXPR_CXX20 bool test_vector_bool(std::size_t N) {
  std::vector<bool> in(N, false);
  for (std::size_t i = 0; i < N; i += 2)
    in[i] = true;

  { // Test copy_backward with aligned bytes
    std::vector<bool> out(N);
    std::copy_backward(in.begin(), in.end(), out.end());
    assert(in == out);
  }
  { // Test copy_backward with unaligned bytes
    std::vector<bool> out(N + 8);
    std::copy_backward(in.begin(), in.end(), out.end() - 4);
    for (std::size_t i = 0; i < N; ++i)
      assert(out[i + 4] == in[i]);
  }

  return true;
};

TEST_CONSTEXPR_CXX20 bool test() {
  test_copy_backward<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
  test_copy_backward<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
  test_copy_backward<bidirectional_iterator<const int*>, int*>();

  test_copy_backward<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
  test_copy_backward<random_access_iterator<const int*>, random_access_iterator<int*> >();
  test_copy_backward<random_access_iterator<const int*>, int*>();

  test_copy_backward<const int*, bidirectional_iterator<int*> >();
  test_copy_backward<const int*, random_access_iterator<int*> >();
  test_copy_backward<const int*, int*>();

#if TEST_STD_VER > 17
  test_copy_backward<contiguous_iterator<const int*>, bidirectional_iterator<int*>>();
  test_copy_backward<contiguous_iterator<const int*>, random_access_iterator<int*>>();
  test_copy_backward<contiguous_iterator<const int*>, int*>();

  test_copy_backward<bidirectional_iterator<const int*>, contiguous_iterator<int*>>();
  test_copy_backward<random_access_iterator<const int*>, contiguous_iterator<int*>>();
  test_copy_backward<contiguous_iterator<const int*>, contiguous_iterator<int*>>();
  test_copy_backward<const int*, contiguous_iterator<int*>>();
#endif

  { // Make sure that padding bits aren't copied
    Derived src(1, 2, 3);
    Derived dst(4, 5, 6);
    std::copy_backward(
        static_cast<PaddedBase*>(&src), static_cast<PaddedBase*>(&src) + 1, static_cast<PaddedBase*>(&dst) + 1);
    assert(dst.a_ == 1);
    assert(dst.b_ == 2);
    assert(dst.c_ == 6);
  }

  { // Make sure that overlapping ranges can be copied
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::copy_backward(a, a + 7, a + 10);
    int expected[] = {1, 2, 3, 1, 2, 3, 4, 5, 6, 7};
    assert(std::equal(a, a + 10, expected));
  }

  { // Test vector<bool>::iterator optimization
    assert(test_vector_bool(8));
    assert(test_vector_bool(16));
    assert(test_vector_bool(32));
    assert(test_vector_bool(64));
    assert(test_vector_bool(256));
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
