//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class I2>
//   requires assignable_from<I&, const I2&>
//     constexpr counted_iterator& operator=(const counted_iterator<I2>& x);

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

class AssignableFromIter
{
    int *it_;

public:
    typedef          std::input_iterator_tag                      iterator_category;
    typedef int                                                   value_type;
    typedef typename std::iterator_traits<int *>::difference_type difference_type;
    typedef int *                                                 pointer;
    typedef int &                                                 reference;

    friend constexpr int *base(const AssignableFromIter& i) {return i.it_;}

    AssignableFromIter() = default;
    explicit constexpr AssignableFromIter(int *it) : it_(it) {}
    constexpr AssignableFromIter(const forward_iterator<int*>& it) : it_(base(it)) {}

    constexpr AssignableFromIter& operator=(const forward_iterator<int*> &other) {
      it_ = base(other);
      return *this;
    }

    constexpr reference operator*() const {return *it_;}

    constexpr AssignableFromIter& operator++() {++it_; return *this;}
    constexpr AssignableFromIter operator++(int)
        {AssignableFromIter tmp(*this); ++(*this); return tmp;}
};

struct InputOrOutputArchetype {
  using difference_type = int;

  int *ptr;

  int operator*() { return *ptr; }
  void operator++(int) { ++ptr; }
  InputOrOutputArchetype& operator++() { ++ptr; return *this; }
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( std::is_assignable_v<std::counted_iterator<forward_iterator<int*>>,
                                        std::counted_iterator<forward_iterator<int*>>>);
    static_assert(!std::is_assignable_v<std::counted_iterator<forward_iterator<int*>>,
                                        std::counted_iterator<random_access_iterator<int*>>>);
  }

  {
    std::counted_iterator iter1(AssignableFromIter{buffer}, 8);
    std::counted_iterator iter2(forward_iterator<int*>{buffer + 2}, 6);
    assert(base(iter1.base()) == buffer);
    assert(iter1.count() == 8);
    std::counted_iterator<AssignableFromIter>& result = (iter1 = iter2);
    assert(&result == &iter1);
    assert(base(iter1.base()) == buffer + 2);
    assert(iter1.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter1 = iter2), std::counted_iterator<AssignableFromIter>&);
  }
  {
    std::counted_iterator iter1(AssignableFromIter{buffer}, 8);
    const std::counted_iterator iter2(forward_iterator<int*>{buffer + 2}, 6);
    assert(base(iter1.base()) == buffer);
    assert(iter1.count() == 8);
    std::counted_iterator<AssignableFromIter>& result = (iter1 = iter2);
    assert(&result == &iter1);
    assert(base(iter1.base()) == buffer + 2);
    assert(iter1.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter1 = iter2), std::counted_iterator<AssignableFromIter>&);
  }

  {
    std::counted_iterator iter1(InputOrOutputArchetype{buffer}, 8);
    std::counted_iterator iter2(InputOrOutputArchetype{buffer + 2}, 6);
    assert(iter1.base().ptr == buffer);
    assert(iter1.count() == 8);
    std::counted_iterator<InputOrOutputArchetype>& result = (iter1 = iter2);
    assert(&result == &iter1);
    assert(iter1.base().ptr == buffer + 2);
    assert(iter1.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter1 = iter2), std::counted_iterator<InputOrOutputArchetype>&);
  }
  {
    std::counted_iterator iter1(InputOrOutputArchetype{buffer}, 8);
    const std::counted_iterator iter2(InputOrOutputArchetype{buffer + 2}, 6);
    assert(iter1.base().ptr == buffer);
    assert(iter1.count() == 8);
    std::counted_iterator<InputOrOutputArchetype>& result = (iter1 = iter2);
    assert(&result == &iter1);
    assert(iter1.base().ptr == buffer + 2);
    assert(iter1.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter1 = iter2), std::counted_iterator<InputOrOutputArchetype>&);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
