//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::next

#include <iterator>

#include "test_standard_function.h"

static_assert(is_function_like<decltype(std::ranges::next)>());

// FIXME: We're bending the rules here by adding a new type to namespace std::ranges. Since this is
// the standard library's test suite, this should be fine (we *are* the implementation), but it's
// necessary at the time of writing since there aren't any iterators in std::ranges that we can
// borrow for this test.
namespace std::ranges {
class fake_forward_iterator {
public:
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;

  fake_forward_iterator() = default;

  value_type operator*() const;
  fake_forward_iterator& operator++();
  fake_forward_iterator operator++(int);

  bool operator==(fake_forward_iterator const&) const = default;
};
} // namespace std::ranges

// The function templates defined in [range.iter.ops] are not found by argument-dependent name lookup ([basic.lookup.argdep]).
template <class I, class... Args>
constexpr bool unqualified_lookup_works = requires(I i, Args... args) {
  next(i, args...);
};

static_assert(!unqualified_lookup_works<std::ranges::fake_forward_iterator>);
static_assert(!unqualified_lookup_works<std::ranges::fake_forward_iterator, std::ptrdiff_t>);
static_assert(!unqualified_lookup_works<std::ranges::fake_forward_iterator, std::ranges::fake_forward_iterator>);
static_assert(
    !unqualified_lookup_works<std::ranges::fake_forward_iterator, std::ptrdiff_t, std::ranges::fake_forward_iterator>);

namespace test {
template <class>
class forward_iterator {
public:
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;

  forward_iterator() = default;

  value_type operator*() const;
  forward_iterator& operator++();
  forward_iterator operator++(int);

  bool operator==(forward_iterator const&) const = default;
};

template <class I>
void next(forward_iterator<I>) {
  static_assert(std::same_as<I, I*>);
}

template <class I>
void next(forward_iterator<I>, std::ptrdiff_t) {
  static_assert(std::same_as<I, I*>);
}

template <class I>
void next(forward_iterator<I>, forward_iterator<I>) {
  static_assert(std::same_as<I, I*>);
}

template <class I>
void next(forward_iterator<I>, std::ptrdiff_t, forward_iterator<I>) {
  static_assert(std::same_as<I, I*>);
}
} // namespace test

// When found by unqualified ([basic.lookup.unqual]) name lookup for the postfix-expression in a
// function call ([expr.call]), they inhibit argument-dependent name lookup.
void adl_inhibition() {
  test::forward_iterator<int*> x;

  using std::ranges::next;

  (void)next(x);
  (void)next(x, 5);
  (void)next(x, x);
  (void)next(x, 6, x);
}
