//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>
//
// reverse_iterator
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator==(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator!=(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator<(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator>(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator<=(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator>=(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template<class Iterator1, three_way_comparable_with<Iterator1> Iterator2>
//  constexpr compare_three_way_result_t<Iterator1, Iterator2>
//    operator<=>(const reverse_iterator<Iterator1>& x,
//                const reverse_iterator<Iterator2>& y);

#include <iterator>
#include <cassert>

#include "test_macros.h"

struct IterBase {
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = int;
  using difference_type = ptrdiff_t;
  using pointer = int*;
  using reference = int&;

  reference operator*() const;
  pointer operator->() const;
};

template<class T> concept HasEqual = requires (T t) { t == t; };
template<class T> concept HasNotEqual = requires (T t) { t != t; };
template<class T> concept HasLess = requires (T t) { t < t; };
template<class T> concept HasLessOrEqual = requires (T t) { t <= t; };
template<class T> concept HasGreater = requires (T t) { t > t; };
template<class T> concept HasGreaterOrEqual = requires (T t) { t >= t; };
template<class T> concept HasSpaceship = requires (T t) { t <=> t; };

// operator ==

struct NoEqualityCompIter : IterBase {
  bool operator!=(NoEqualityCompIter) const;
  bool operator<(NoEqualityCompIter) const;
  bool operator>(NoEqualityCompIter) const;
  bool operator<=(NoEqualityCompIter) const;
  bool operator>=(NoEqualityCompIter) const;
};

static_assert( HasEqual<std::reverse_iterator<int*>>);
static_assert(!HasEqual<std::reverse_iterator<NoEqualityCompIter>>);
static_assert( HasNotEqual<std::reverse_iterator<NoEqualityCompIter>>);
static_assert( HasLess<std::reverse_iterator<NoEqualityCompIter>>);
static_assert( HasLessOrEqual<std::reverse_iterator<NoEqualityCompIter>>);
static_assert( HasGreater<std::reverse_iterator<NoEqualityCompIter>>);
static_assert( HasGreaterOrEqual<std::reverse_iterator<NoEqualityCompIter>>);

void Foo() {
  std::reverse_iterator<NoEqualityCompIter> i;
  (void)i;
}

// operator !=

struct NoInequalityCompIter : IterBase {
  bool operator<(NoInequalityCompIter) const;
  bool operator>(NoInequalityCompIter) const;
  bool operator<=(NoInequalityCompIter) const;
  bool operator>=(NoInequalityCompIter) const;
};

static_assert( HasNotEqual<std::reverse_iterator<int*>>);
static_assert(!HasNotEqual<std::reverse_iterator<NoInequalityCompIter>>);
static_assert(!HasEqual<std::reverse_iterator<NoInequalityCompIter>>);
static_assert( HasLess<std::reverse_iterator<NoInequalityCompIter>>);
static_assert( HasLessOrEqual<std::reverse_iterator<NoInequalityCompIter>>);
static_assert( HasGreater<std::reverse_iterator<NoInequalityCompIter>>);
static_assert( HasGreaterOrEqual<std::reverse_iterator<NoInequalityCompIter>>);

// operator <

struct NoGreaterCompIter : IterBase {
  bool operator==(NoGreaterCompIter) const;
  bool operator!=(NoGreaterCompIter) const;
  bool operator<(NoGreaterCompIter) const;
  bool operator<=(NoGreaterCompIter) const;
  bool operator>=(NoGreaterCompIter) const;
};

static_assert( HasLess<std::reverse_iterator<int*>>);
static_assert(!HasLess<std::reverse_iterator<NoGreaterCompIter>>);
static_assert( HasEqual<std::reverse_iterator<NoGreaterCompIter>>);
static_assert( HasNotEqual<std::reverse_iterator<NoGreaterCompIter>>);
static_assert( HasLessOrEqual<std::reverse_iterator<NoGreaterCompIter>>);
static_assert( HasGreater<std::reverse_iterator<NoGreaterCompIter>>);
static_assert( HasGreaterOrEqual<std::reverse_iterator<NoGreaterCompIter>>);

// operator >

struct NoLessCompIter : IterBase {
  bool operator==(NoLessCompIter) const;
  bool operator!=(NoLessCompIter) const;
  bool operator>(NoLessCompIter) const;
  bool operator<=(NoLessCompIter) const;
  bool operator>=(NoLessCompIter) const;
};

static_assert( HasGreater<std::reverse_iterator<int*>>);
static_assert(!HasGreater<std::reverse_iterator<NoLessCompIter>>);
static_assert( HasEqual<std::reverse_iterator<NoLessCompIter>>);
static_assert( HasNotEqual<std::reverse_iterator<NoLessCompIter>>);
static_assert( HasLess<std::reverse_iterator<NoLessCompIter>>);
static_assert( HasLessOrEqual<std::reverse_iterator<NoLessCompIter>>);
static_assert( HasGreaterOrEqual<std::reverse_iterator<NoLessCompIter>>);

// operator <=

struct NoGreaterOrEqualCompIter : IterBase {
  bool operator==(NoGreaterOrEqualCompIter) const;
  bool operator!=(NoGreaterOrEqualCompIter) const;
  bool operator<(NoGreaterOrEqualCompIter) const;
  bool operator>(NoGreaterOrEqualCompIter) const;
  bool operator<=(NoGreaterOrEqualCompIter) const;
};

static_assert( HasLessOrEqual<std::reverse_iterator<int*>>);
static_assert(!HasLessOrEqual<std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert( HasEqual<std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert( HasNotEqual<std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert( HasLess<std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert( HasGreater<std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert( HasGreaterOrEqual<std::reverse_iterator<NoGreaterOrEqualCompIter>>);

// operator >=

struct NoLessOrEqualCompIter : IterBase {
  bool operator==(NoLessOrEqualCompIter) const;
  bool operator!=(NoLessOrEqualCompIter) const;
  bool operator<(NoLessOrEqualCompIter) const;
  bool operator>(NoLessOrEqualCompIter) const;
  bool operator>=(NoLessOrEqualCompIter) const;
};

static_assert( HasGreaterOrEqual<std::reverse_iterator<int*>>);
static_assert(!HasGreaterOrEqual<std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert( HasEqual<std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert( HasNotEqual<std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert( HasLess<std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert( HasLessOrEqual<std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert( HasGreater<std::reverse_iterator<NoLessOrEqualCompIter>>);

// operator <=>

static_assert( std::three_way_comparable_with<int*, int*>);
static_assert( HasSpaceship<std::reverse_iterator<int*>>);
static_assert(!std::three_way_comparable_with<NoEqualityCompIter, NoEqualityCompIter>);
static_assert(!HasSpaceship<std::reverse_iterator<NoEqualityCompIter>>);
static_assert(!std::three_way_comparable_with<NoInequalityCompIter, NoInequalityCompIter>);
static_assert(!HasSpaceship<std::reverse_iterator<NoInequalityCompIter>>);
static_assert(!std::three_way_comparable_with<NoGreaterCompIter, NoGreaterCompIter>);
static_assert(!HasSpaceship<std::reverse_iterator<NoGreaterCompIter>>);
static_assert(!std::three_way_comparable_with<NoLessCompIter, NoLessCompIter>);
static_assert(!HasSpaceship<std::reverse_iterator<NoLessCompIter>>);
static_assert(!std::three_way_comparable_with<NoGreaterOrEqualCompIter, NoGreaterOrEqualCompIter>);
static_assert(!HasSpaceship<std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert(!std::three_way_comparable_with<NoLessOrEqualCompIter, NoLessOrEqualCompIter>);
static_assert(!HasSpaceship<std::reverse_iterator<NoLessOrEqualCompIter>>);
