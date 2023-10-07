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

#include "test_macros.h"
#include "test_iterators.h"
#include "almost_satisfies_types.h"

// Concepts to check different overloads of std::ranges::iota
template <class Iter = int*, class Sent = int*, class Value = int>
concept HasIotaIter = requires(Iter&& iter, Sent&& sent, Value&& val) {
  std::ranges::iota(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Value>(val));
};

template <class Range, class Value = int>
concept HasIotaRange = requires(Range&& range, Value&& val) {
  std::ranges::iota(std::forward<Range>(range), std::forward<Value>(val));
};

constexpr void test_constraints() {
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
}

template <class Iter, class Sent, std::size_t N>
constexpr void test_result(std::array<int, N> input, int starting_value, std::array<int, N> const expected) {
  { // (iterator, sentinel) overload
    auto in_begin = Iter(input.data());
    auto in_end   = Sent(Iter(input.data() + input.size()));
    std::same_as<std::ranges::out_value_result<Iter, int>> decltype(auto) result =
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
  if constexpr (!std::is_same_v<Iter, cpp17_input_iterator<int*>> &&
                !std::is_same_v<Iter, cpp20_input_iterator<int*>>) { // (range) overload
    auto in_begin = Iter(input.data());
    auto in_end   = Sent(Iter(input.data() + input.size()));
    auto range    = std::ranges::subrange(std::move(in_begin), std::move(in_end));

    std::same_as<std::ranges::out_value_result<Iter, int>> decltype(auto) result =
        std::ranges::iota(range, starting_value);
    assert(result.out == in_end);
    if constexpr (expected.size() > 0) {
      assert(result.value == expected.back() + 1);
    } else {
      assert(result.value == starting_value);
    }
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

void test_results() {
  test_results<cpp17_input_iterator<int*>>();
  test_results<cpp20_input_iterator<int*>>();
  test_results<cpp17_output_iterator<int*>>();
  test_results<cpp20_output_iterator<int*>>();
  test_results<forward_iterator<int*>>();
  test_results<bidirectional_iterator<int*>>();
  test_results<random_access_iterator<int*>>();
  test_results<contiguous_iterator<int*>>();
  test_results<int*>();
}

int main(int, char**) {
  test_constraints();
  test_results();
  return 0;
}