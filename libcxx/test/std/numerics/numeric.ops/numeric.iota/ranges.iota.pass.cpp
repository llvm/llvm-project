//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Testing std::ranges::iota

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_macros.h"

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

  /*
  This class has a "mischievous" non-const overload of copy-assignment
  operator that modifies the object being assigned from. `ranges::iota`
  should not be invoking this overload thanks to the `std::as_const` in its
  implementation. If for some reason it does invoke it, there will be a compiler
  error.
  */
  constexpr DangerousCopyAssign& operator=(DangerousCopyAssign& a) = delete;

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

template <class Iter, class Sent, std::size_t N>
constexpr void test_result(std::array<int, N> input, int starting_value, std::array<int, N> const expected) {
  { // (iterator, sentinel) overload
    auto in_begin = Iter(input.data());
    auto in_end   = Sent(Iter(input.data() + input.size()));
    std::same_as<std::ranges::out_value_result<Iter, int>> decltype(auto) result =
        std::ranges::iota(std::move(in_begin), std::move(in_end), starting_value);
    assert(result.out == in_end);
    assert(result.value == starting_value + static_cast<int>(N));
    assert(std::ranges::equal(input, expected));
  }

  { // (range) overload
    // in the range overload adds the additional constraint that it must be an output range
    // so skip this for the input iterators we test
    auto in_begin = Iter(input.data());
    auto in_end   = Sent(Iter(input.data() + input.size()));
    auto range    = std::ranges::subrange(std::move(in_begin), std::move(in_end));

    std::same_as<std::ranges::out_value_result<Iter, int>> decltype(auto) result =
        std::ranges::iota(range, starting_value);
    assert(result.out == in_end);
    assert(result.value == starting_value + static_cast<int>(N));
    assert(std::ranges::equal(input, expected));
  }
}

template <class Iter, class Sent = sentinel_wrapper<Iter>>
constexpr void test_results() {
  // Empty
  test_result<Iter, Sent, 0>({}, 0, {});
  // 1-element sequence
  test_result<Iter, Sent, 1>({1}, 0, {0});
  // Longer sequence
  test_result<Iter, Sent, 5>({1, 2, 3, 4, 5}, 0, {0, 1, 2, 3, 4});
}

constexpr void test_user_defined_type() {
  // Simple non-fundamental type
  struct UserDefinedType {
    int val;
    using difference_type = int;

    constexpr explicit UserDefinedType(int v) : val(v) {}
    constexpr UserDefinedType(UserDefinedType const& other) { this->val = other.val; }
    constexpr UserDefinedType& operator=(UserDefinedType const& a) {
      this->val = a.val;
      return *this;
    }

    // prefix
    constexpr UserDefinedType& operator++() {
      ++(this->val);
      return *this;
    }

    // postfix
    constexpr UserDefinedType operator++(int) {
      auto tmp = *this;
      ++this->val;
      return tmp;
    }
  };

  // Setup
  using A                                 = UserDefinedType;
  std::array<UserDefinedType, 5> a        = {A{0}, A{0}, A{0}, A{0}, A{0}};
  std::array<UserDefinedType, 5> expected = {A{0}, A{1}, A{2}, A{3}, A{4}};

  // Fill with values
  std::ranges::iota(a, A{0});
  auto proj_val = [](UserDefinedType const& el) { return el.val; };

  // Check
  assert(std::ranges::equal(a, expected, std::ranges::equal_to{}, proj_val, proj_val));
}

constexpr void test_dangerous_copy_assign() {
  using A = DangerousCopyAssign;

  // If the dangerous non-const copy assignment is called, the final values in
  // aa should increment by 2 rather than 1.
  std::array<A, 3> aa       = {A{0}, A{0}, A{0}};
  std::array<A, 3> expected = {A{0}, A{1}, A{2}};
  std::ranges::iota(aa, A{0});
  auto proj_val = [](DangerousCopyAssign const& el) { return el.val; };
  assert(std::ranges::equal(aa, expected, std::ranges::equal_to{}, proj_val, proj_val));
}

constexpr bool test_results() {
  // Tests on fundamental types
  types::for_each(types::cpp17_input_iterator_list<int*>{}, []<class Iter> { test_results< Iter>(); });
  test_results<cpp17_output_iterator<int*>>();
  test_results<cpp20_output_iterator<int*>>();
  test_results<int*, sized_sentinel<int*>>();

  // Tests on non-fundamental types
  test_user_defined_type();
  test_dangerous_copy_assign();
  return true;
}

int main(int, char**) {
  test_results();
  static_assert(test_results());
  return 0;
}
