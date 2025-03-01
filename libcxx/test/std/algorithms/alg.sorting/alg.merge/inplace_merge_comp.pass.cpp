//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class BidirectionalIterator, class Compare>
//   constexpr void                                         // constexpr since C++26
//   inplace_merge(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, Compare comp);

#include <algorithm>
#include <cassert>
#include <functional>
#include <random>
#include <vector>

#include "test_macros.h"

#if TEST_STD_VER >= 11
#include <memory>

struct indirect_less {
  template <class P>
  TEST_CONSTEXPR_CXX26 bool operator()(const P& x, const P& y) const {
    return *x < *y;
  }
};

struct S {
    TEST_CONSTEXPR_CXX26 S() : i_(0) {}
    TEST_CONSTEXPR_CXX26 S(int i) : i_(i) {}

    TEST_CONSTEXPR_CXX26 S(const S&  rhs) : i_(rhs.i_) {}
    TEST_CONSTEXPR_CXX26 S(      S&& rhs) : i_(rhs.i_) { rhs.i_ = -1; }

    TEST_CONSTEXPR_CXX26 S& operator =(const S&  rhs) { i_ = rhs.i_;              return *this; }
    TEST_CONSTEXPR_CXX26 S& operator =(      S&& rhs) { i_ = rhs.i_; rhs.i_ = -2; assert(this != &rhs); return *this; }
    TEST_CONSTEXPR_CXX26 S& operator =(int i)         { i_ = i;                   return *this; }

    TEST_CONSTEXPR_CXX26 bool operator  <(const S&  rhs) const { return i_ < rhs.i_; }
    TEST_CONSTEXPR_CXX26 bool operator  >(const S&  rhs) const { return i_ > rhs.i_; }
    TEST_CONSTEXPR_CXX26 bool operator ==(const S&  rhs) const { return i_ == rhs.i_; }
    TEST_CONSTEXPR_CXX26 bool operator ==(int i)         const { return i_ == i; }

    TEST_CONSTEXPR_CXX26 void set(int i) { i_ = i; }

    int i_;
    };


#endif // TEST_STD_VER >= 11

#include "test_iterators.h"
#include "counting_predicates.h"

std::mt19937 randomness;

template <class Iter>
void
test_one_randomized(unsigned N, unsigned M)
{
    typedef typename std::iterator_traits<Iter>::value_type value_type;
    value_type* ia = new value_type[N];
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    std::shuffle(ia, ia+N, randomness);
    std::sort(ia, ia+M, std::greater<value_type>());
    std::sort(ia+M, ia+N, std::greater<value_type>());
    binary_counting_predicate<std::greater<value_type>, value_type, value_type> pred((std::greater<value_type>()));
    std::inplace_merge(Iter(ia), Iter(ia+M), Iter(ia+N), std::ref(pred));
    if(N > 0)
    {
        assert(ia[0] == static_cast<int>(N)-1);
        assert(ia[N-1] == 0);
        assert(std::is_sorted(ia, ia+N, std::greater<value_type>()));
#if defined(_LIBCPP_HARDENING_MODE) && _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_DEBUG
        assert(pred.count() <= (N-1));
#endif
    }
    delete [] ia;
}

template <class Iter>
TEST_CONSTEXPR_CXX26 void test_one_non_randomized(unsigned N, unsigned M) {
  typedef typename std::iterator_traits<Iter>::value_type value_type;

  value_type* ia                  = new value_type[N];
  const unsigned long small_prime = 19937;
  const unsigned long large_prime = 212987;
  unsigned long product_mod       = small_prime;
  for (unsigned i = 0; i < N; ++i) {
    ia[i]       = static_cast<int>(product_mod);
    product_mod = product_mod * small_prime % large_prime;
  }
  std::sort(ia, ia + M, std::greater<value_type>());
  std::sort(ia + M, ia + N, std::greater<value_type>());
  binary_counting_predicate<std::greater<value_type>, value_type, value_type> pred((std::greater<value_type>()));
  std::inplace_merge(Iter(ia), Iter(ia + M), Iter(ia + N), std::ref(pred));
  if (N > 0) {
    assert(std::is_sorted(ia, ia + N, std::greater<value_type>()));
#if defined(_LIBCPP_HARDENING_MODE) && _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_DEBUG
    assert(pred.count() <= (N - 1));
#endif
  }
  delete[] ia;
}

template <class Iter>
TEST_CONSTEXPR_CXX26 void test_one(unsigned N, unsigned M) {
  assert(M <= N);
  if (!TEST_IS_CONSTANT_EVALUATED) {
    test_one_randomized<Iter>(N, M);
  }
  test_one_non_randomized<Iter>(N, M);
}

template <class Iter>
TEST_CONSTEXPR_CXX26 void
test(unsigned N)
{
    test_one<Iter>(N, 0);
    test_one<Iter>(N, N/4);
    test_one<Iter>(N, N/2);
    test_one<Iter>(N, 3*N/4);
    test_one<Iter>(N, N);
}

template <class Iter>
TEST_CONSTEXPR_CXX26 void
test()
{
    test_one<Iter>(0, 0);
    test_one<Iter>(1, 0);
    test_one<Iter>(1, 1);
    test_one<Iter>(2, 0);
    test_one<Iter>(2, 1);
    test_one<Iter>(2, 2);
    test_one<Iter>(3, 0);
    test_one<Iter>(3, 1);
    test_one<Iter>(3, 2);
    test_one<Iter>(3, 3);
    test<Iter>(4);
    test<Iter>(20);
#if defined(_LIBCPP_HARDENING_MODE)
    if (!TEST_IS_CONSTANT_EVALUATED) // avoid blowing past constant evaluation limit
#endif
    {
      test<Iter>(100);
    }
    if (!TEST_IS_CONSTANT_EVALUATED) { // avoid blowing past constant evaluation limit
      test<Iter>(1000);
    }
}

struct less_by_first {
  template <typename Pair>
  TEST_CONSTEXPR_CXX26 bool operator()(const Pair& lhs, const Pair& rhs) const {
    return std::less<typename Pair::first_type>()(lhs.first, rhs.first);
  }
};

TEST_CONSTEXPR_CXX26 void test_PR31166()
{
    typedef std::pair<int, int> P;
    typedef std::vector<P> V;
    P vec[5] = {P(1, 0), P(2, 0), P(2, 1), P(2, 2), P(2, 3)};
    for ( int i = 0; i < 5; ++i ) {
        V res(vec, vec + 5);
        std::inplace_merge(res.begin(), res.begin() + i, res.end(), less_by_first());
        assert(res.size() == 5);
        assert(std::equal(res.begin(), res.end(), vec));
    }
}

#if TEST_STD_VER >= 11
void test_wrapped_randomized(int N, unsigned M) {
  std::unique_ptr<int>* ia = new std::unique_ptr<int>[N];
  for (int i = 0; i < N; ++i)
    ia[i].reset(new int(i));
  std::shuffle(ia, ia + N, randomness);
  std::sort(ia, ia + M, indirect_less());
  std::sort(ia + M, ia + N, indirect_less());
  std::inplace_merge(ia, ia + M, ia + N, indirect_less());
  if (N > 0) {
    assert(*ia[0] == 0);
    assert(*ia[N - 1] == N - 1);
    assert(std::is_sorted(ia, ia + N, indirect_less()));
  }
  delete[] ia;
}

TEST_CONSTEXPR_CXX26 void test_wrapped_non_randomized(int N, unsigned M) {
  std::unique_ptr<int>* ia = new std::unique_ptr<int>[N];

  const unsigned long small_prime = 19937;
  const unsigned long large_prime = 212987;
  unsigned long product_mod       = small_prime;
  for (int i = 0; i < N; ++i) {
    ia[i].reset(new int(static_cast<int>(product_mod)));
    product_mod = product_mod * small_prime % large_prime;
  }
  std::sort(ia, ia + M, indirect_less());
  std::sort(ia + M, ia + N, indirect_less());
  std::inplace_merge(ia, ia + M, ia + N, indirect_less());
  if (N > 0) {
    assert(std::is_sorted(ia, ia + N, indirect_less()));
  }
  delete[] ia;
}
#endif // TEST_STD_VER >= 11

TEST_CONSTEXPR_CXX26 bool test()
{
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();

#if TEST_STD_VER >= 11
    test<bidirectional_iterator<S*> >();
    test<random_access_iterator<S*> >();
    test<S*>();

    if (!TEST_IS_CONSTANT_EVALUATED) {
      test_wrapped_randomized(100, 50);
    }
    test_wrapped_non_randomized(100, 50);
#endif // TEST_STD_VER >= 11

    test_PR31166();

    return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif // TEST_STD_VER >= 26

  return 0;
}
