//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// constexpr counted_iterator& operator++();
// decltype(auto) operator++(int);
// constexpr counted_iterator operator++(int)
//   requires forward_iterator<I>;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
template <class It>
class ThrowsOnInc
{
    It it_;

public:
    typedef          std::input_iterator_tag                   iterator_category;
    typedef typename std::iterator_traits<It>::value_type      value_type;
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename std::iterator_traits<It>::reference       reference;

    constexpr It base() const {return it_;}

    ThrowsOnInc() = default;
    explicit constexpr ThrowsOnInc(It it) : it_(it) {}

    constexpr reference operator*() const {return *it_;}

    constexpr ThrowsOnInc& operator++() {throw 42;}
    constexpr ThrowsOnInc operator++(int) {throw 42;}
};
#endif // TEST_HAS_NO_EXCEPTIONS

struct InputOrOutputArchetype {
  using difference_type = int;

  int *ptr;

  constexpr int operator*() const { return *ptr; }
  constexpr void operator++(int) { ++ptr; }
  constexpr InputOrOutputArchetype& operator++() { ++ptr; return *this; }
};

template<class Iter>
concept PlusEnabled = requires(Iter& iter) {
  iter++;
  ++iter;
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using Counted = std::counted_iterator<forward_iterator<int*>>;
    std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);

    assert(iter++ == Counted(forward_iterator<int*>{buffer}, 8));
    assert(++iter == Counted(forward_iterator<int*>{buffer + 2}, 6));

    ASSERT_SAME_TYPE(decltype(iter++), Counted);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }
  {
    using Counted = std::counted_iterator<random_access_iterator<int*>>;
    std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);

    assert(iter++ == Counted(random_access_iterator<int*>{buffer}, 8));
    assert(++iter == Counted(random_access_iterator<int*>{buffer + 2}, 6));

    ASSERT_SAME_TYPE(decltype(iter++), Counted);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }

  {
    static_assert( PlusEnabled<      std::counted_iterator<random_access_iterator<int*>>>);
    static_assert(!PlusEnabled<const std::counted_iterator<random_access_iterator<int*>>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using Counted = std::counted_iterator<InputOrOutputArchetype>;
    std::counted_iterator iter(InputOrOutputArchetype{buffer}, 8);

    iter++;
    assert((++iter).base().ptr == buffer + 2);

    ASSERT_SAME_TYPE(decltype(iter++), void);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }
  {
    using Counted = std::counted_iterator<cpp20_input_iterator<int*>>;
    std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);

    iter++;
    assert(++iter == Counted(cpp20_input_iterator<int*>{buffer + 2}, 6));

    ASSERT_SAME_TYPE(decltype(iter++), void);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using Counted = std::counted_iterator<ThrowsOnInc<int*>>;
    std::counted_iterator iter(ThrowsOnInc<int*>{buffer}, 8);
    try {
      (void)iter++;
      assert(false);
    } catch (int x) {
      assert(x == 42);
      assert(iter.count() == 8);
    }

    ASSERT_SAME_TYPE(decltype(iter++), ThrowsOnInc<int*>);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  return 0;
}
