//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Testing std::ranges::iota

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <numeric>
#include <algorithm>
#include <array>
#include <iostream> // TODO RM

#include "test_macros.h"
#include "test_iterators.h"
#include "almost_satisfies_types.h"

//
// Testing constraints
//

// Concepts to check different overloads of std::ranges::iota
template <class Iter = int*, class Sent = int*, class Value = int>
concept HasIotaIter = requires(Iter&& iter, Sent&& sent, Value&& val) {
  std::ranges::iota(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Value>(val));
};

template <class Range, class Value = int>
concept HasIotaRange =
    requires(Range&& range, Value&& val) { std::ranges::iota(std::forward<Range>(range), std::forward<Value>(val)); };

// Test constraints of the iterator/sentinel overload
// ==================================================
static_assert(HasIotaIter<int*, int*, int>);

// !input_or_output_iterator<O>
static_assert(!HasIotaIter<InputIteratorNotInputOrOutputIterator>);

// !sentinel_for<S, O>
static_assert(!HasIotaIter<int*, SentinelForNotSemiregular>);
static_assert(!HasIotaIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !weakly_incrementable<T>
static_assert(!HasIotaIter<int*, int*, WeaklyIncrementableNotMovable>);

// !indirectly writable <O, T>
static_assert(!HasIotaIter<OutputIteratorNotIndirectlyWritable, int*, int>);

// Test constraints for the range overload
// =======================================
static_assert(HasIotaRange<UncheckedRange<int*>, int>);

// !weakly_incrementable<T>
static_assert(!HasIotaRange<UncheckedRange<int*>, WeaklyIncrementableNotMovable>);

// !ranges::output_range<const _Tp&>
static_assert(!HasIotaRange<UncheckedRange<int*>, OutputIteratorNotIndirectlyWritable>);

//
// Testing results
//

struct DangerousCopyAssign {
  int val;
  using difference_type = int;

  constexpr explicit DangerousCopyAssign(int v) : val(v) {}

  // Needed in postfix
  constexpr DangerousCopyAssign(DangerousCopyAssign const& other) { this->val = other.val; }

  // mischievious copy assignment that we won't use if the
  // std::as_const inside ranges::iota isn't working, this should perturb the
  // results
  constexpr DangerousCopyAssign& operator=(DangerousCopyAssign& a) {
    ++a.val;
    this->val = a.val;
    return *this;
  }

  // safe copy assignment std::as_const inside ranges::iota should ensure this
  // overload gets called
  constexpr DangerousCopyAssign& operator=(DangerousCopyAssign const& a) {
    this->val = a.val;
    return *this;
  }

  constexpr bool operator==(DangerousCopyAssign const& rhs) { return this->val == rhs.val; }

  // prefix
  constexpr DangerousCopyAssign& operator++() {
    ++(this->val);
    return *this;
  }

  // postfix
  constexpr DangerousCopyAssign operator++(int) {
    auto tmp = *this;
    ++this->val;
    return tmp;
  }
};

template <class T, class Iter, class Sent, std::size_t N>
constexpr void test_result(std::array<T, N> input, T starting_value, std::array<T, N> const expected) {
  { // (iterator, sentinel) overload
    auto in_begin = Iter(input.data());
    auto in_end   = Sent(Iter(input.data() + input.size()));
    std::same_as<std::ranges::out_value_result<Iter, T>> decltype(auto) result =
        std::ranges::iota(std::move(in_begin), std::move(in_end), starting_value);
    assert(result.out == in_end);
    if constexpr (expected.size() > 0) {
      assert(result.value == expected.back() + 1);
    } else {
      assert(result.value == starting_value);
    }
    assert(std::ranges::equal(input, expected));
  }

  // The range overload adds the additional constraint that it must be an outputrange
  // so skip this for the input iterators we test
  if constexpr (!std::is_same_v<Iter, cpp17_input_iterator<T*>> &&
                !std::is_same_v<Iter, cpp20_input_iterator<T*>>) { // (range) overload
    auto in_begin = Iter(input.data());
    auto in_end   = Sent(Iter(input.data() + input.size()));
    auto range    = std::ranges::subrange(std::move(in_begin), std::move(in_end));

    std::same_as<std::ranges::out_value_result<Iter, T>> decltype(auto) result =
        std::ranges::iota(range, starting_value);
    assert(result.out == in_end);
    assert(result.value == starting_value + N);
    assert(std::ranges::equal(input, expected));
  }
}

template <class Iter, class Sent = sentinel_wrapper<Iter>>
constexpr void test_results() {
  // Empty
  test_result<int, Iter, Sent, 0>({}, 0, {});
  // 1-element sequence
  test_result<int, Iter, Sent, 1>({1}, 0, {0});
  // Longer sequence
  test_result<int, Iter, Sent, 5>({1, 2, 3, 4, 5}, 0, {0, 1, 2, 3, 4});
}

constexpr void test_DangerousCopyAssign() {
  using A                   = DangerousCopyAssign;
  using Iter                = contiguous_iterator<A*>;
  std::array<A, 3> aa       = {A{1}, A{2}, A{3}};
  std::array<A, 3> expected = {A{0}, A{1}, A{2}};
  std::ranges::iota(aa, A{0});
  auto proj_val = [](DangerousCopyAssign const& el) { return el.val; };
  assert(std::ranges::equal(aa, expected, std::ranges::equal_to{}, proj_val, proj_val));
}

void test_results() {
  types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iter> { test_results< Iter>(); });
  test_results<cpp17_output_iterator<int*>>();
  test_results<cpp20_output_iterator<int*>>();
  test_DangerousCopyAssign();
}

int main(int, char**) {
  test_results();
  return 0;
}
