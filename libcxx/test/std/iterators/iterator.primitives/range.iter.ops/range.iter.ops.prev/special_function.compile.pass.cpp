//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::prev

#include <iterator>

#include "test_iterators.h"
#include "test_standard_function.h"

static_assert(is_function_like<decltype(std::ranges::prev)>());

namespace std::ranges {
class fake_bidirectional_iterator {
public:
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;

  fake_bidirectional_iterator() = default;

  value_type operator*() const;
  fake_bidirectional_iterator& operator++();
  fake_bidirectional_iterator operator++(int);
  fake_bidirectional_iterator& operator--();
  fake_bidirectional_iterator operator--(int);

  bool operator==(fake_bidirectional_iterator const&) const = default;
};
} // namespace std::ranges

// The function templates defined in [range.iter.ops] are not found by argument-dependent name lookup ([basic.lookup.argdep]).
template <class I, class... Args>
constexpr bool unqualified_lookup_works = requires(I i, Args... args) {
  prev(i, args...);
};

static_assert(!unqualified_lookup_works<std::ranges::fake_bidirectional_iterator>);
static_assert(!unqualified_lookup_works<std::ranges::fake_bidirectional_iterator, std::ptrdiff_t>);
static_assert(!unqualified_lookup_works<std::ranges::fake_bidirectional_iterator, std::ptrdiff_t,
                                        std::ranges::fake_bidirectional_iterator>);

namespace test {
template <class>
class bidirectional_iterator {
public:
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;

  bidirectional_iterator() = default;

  value_type operator*() const;
  bidirectional_iterator& operator++();
  bidirectional_iterator operator++(int);
  bidirectional_iterator& operator--();
  bidirectional_iterator operator--(int);

  bool operator==(bidirectional_iterator const&) const = default;
};

template <class I>
void prev(bidirectional_iterator<I>) {
  static_assert(std::same_as<I, I*>);
}

template <class I>
void prev(bidirectional_iterator<I>, std::ptrdiff_t) {
  static_assert(std::same_as<I, I*>);
}

template <class I>
void prev(bidirectional_iterator<I>, std::ptrdiff_t, bidirectional_iterator<I>) {
  static_assert(std::same_as<I, I*>);
}
} // namespace test

// When found by unqualified ([basic.lookup.unqual]) name lookup for the postfix-expression in a
// function call ([expr.call]), they inhibit argument-dependent name lookup.
void adl_inhibition() {
  test::bidirectional_iterator<int*> x;

  using std::ranges::prev;

  (void)prev(x);
  (void)prev(x, 5);
  (void)prev(x, 6, x);
}
