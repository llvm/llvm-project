//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// Test nested types and data member:

// template <BidirectionalIterator Iter>
// class reverse_iterator {
// protected:
//   Iter current;
// public:
//   iterator<typename iterator_traits<Iterator>::iterator_category,
//   typename iterator_traits<Iterator>::value_type,
//   typename iterator_traits<Iterator>::difference_type,
//   typename iterator_traits<Iterator>::pointer,
//   typename iterator_traits<Iterator>::reference> {
// };

#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
struct find_current
    : private std::reverse_iterator<It>
{
    void test() { (void)this->current; }
};

template <class It>
void test() {
  typedef std::reverse_iterator<It> R;
  typedef std::iterator_traits<It> T;
  find_current<It> q; q.test(); // Just test that we can access `.current` from derived classes
  static_assert((std::is_same<typename R::iterator_type, It>::value), "");
  static_assert((std::is_same<typename R::value_type, typename T::value_type>::value), "");
  static_assert((std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
  static_assert((std::is_same<typename R::reference, typename T::reference>::value), "");
  static_assert((std::is_same<typename R::pointer, typename std::iterator_traits<It>::pointer>::value), "");

#if TEST_STD_VER <= 14
  typedef std::iterator<typename T::iterator_category, typename T::value_type> iterator_base;
  static_assert((std::is_base_of<iterator_base, R>::value), "");
#endif
#if TEST_STD_VER > 17
  if constexpr (std::is_same_v<typename T::iterator_category, std::contiguous_iterator_tag>) {
    static_assert((std::is_same<typename R::iterator_category, std::random_access_iterator_tag>::value), "");
  } else {
    static_assert((std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
  }
#else
  static_assert((std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
#endif
}

#if TEST_STD_VER > 17

struct FooIter {
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = void*;
  using difference_type = void*;
  using pointer = void*;
  using reference = int&;
  int& operator*() const;
};
template <>
struct std::indirectly_readable_traits<FooIter> {
  using value_type = int;
};
template <>
struct std::incrementable_traits<FooIter> {
  using difference_type = char;
};

// Not using `FooIter::value_type`.
static_assert(std::is_same_v<typename std::reverse_iterator<FooIter>::value_type, int>);
// Not using `FooIter::difference_type`.
static_assert(std::is_same_v<typename std::reverse_iterator<FooIter>::difference_type, char>);

#endif

struct BarIter {
  bool& operator*() const;
};
template <>
struct std::iterator_traits<BarIter> {
  using difference_type = char;
  using value_type = char;
  using pointer = char*;
  using reference = char&;
  using iterator_category = std::bidirectional_iterator_tag;
};

#if TEST_STD_VER > 17
  static_assert(std::is_same_v<typename std::reverse_iterator<BarIter>::reference, bool&>);
#else
  static_assert(std::is_same<typename std::reverse_iterator<BarIter>::reference, char&>::value, "");
#endif

void test_all() {
  test<bidirectional_iterator<char*> >();
  test<random_access_iterator<char*> >();
  test<char*>();

#if TEST_STD_VER > 17
  test<contiguous_iterator<char*>>();
  static_assert(std::is_same_v<typename std::reverse_iterator<bidirectional_iterator<char*>>::iterator_concept, std::bidirectional_iterator_tag>);
  static_assert(std::is_same_v<typename std::reverse_iterator<random_access_iterator<char*>>::iterator_concept, std::random_access_iterator_tag>);
  static_assert(std::is_same_v<typename std::reverse_iterator<contiguous_iterator<char*>>::iterator_concept, std::random_access_iterator_tag>);
  static_assert(std::is_same_v<typename std::reverse_iterator<char*>::iterator_concept, std::random_access_iterator_tag>);
#endif
}
