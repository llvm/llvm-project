//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<charT> R>
//   constexpr basic_string(from_range_t, R&& rg, const Allocator& a = Allocator());           // since C++23

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../../containers/from_range_helpers.h"
#include "../../../containers/sequences/from_range_sequence_containers.h"
#include "test_macros.h"
#include "asan_testing.h"

template <class Container, class Range, class Alloc>
concept StringHasFromRangeAllocCtr =
    requires(Range&& range, const Alloc& alloc) { Container(std::from_range, std::forward<Range>(range), alloc); };

constexpr bool test_constraints() {
  // (from_range, range)
  //
  // Input range with the same value type.
  static_assert(HasFromRangeCtr<std::string, InputRange<char>>);
  // Input range with a convertible value type.
  static_assert(HasFromRangeCtr<std::string, InputRange<int>>);
  // Input range with a non-convertible value type.
  static_assert(!HasFromRangeCtr<std::string, InputRange<Empty>>);
  // Not an input range.
  static_assert(!HasFromRangeCtr<std::string, InputRangeNotDerivedFrom>);
  static_assert(!HasFromRangeCtr<std::string, InputRangeNotIndirectlyReadable>);
  static_assert(!HasFromRangeCtr<std::string, InputRangeNotInputOrOutputIterator>);

  // (from_range, range, alloc)
  //
  // Input range with the same value type.
  using Alloc           = test_allocator<char>;
  using StringWithAlloc = std::basic_string<char, std::char_traits<char>, Alloc>;
  static_assert(StringHasFromRangeAllocCtr<StringWithAlloc, InputRange<char>, Alloc>);
  // Input range with a convertible value type.
  static_assert(StringHasFromRangeAllocCtr<StringWithAlloc, InputRange<int>, Alloc>);
  // Input range with a non-convertible value type.
  static_assert(!StringHasFromRangeAllocCtr<StringWithAlloc, InputRange<Empty>, Alloc>);
  // Not an input range.
  static_assert(!StringHasFromRangeAllocCtr<StringWithAlloc, InputRangeNotDerivedFrom, Alloc>);
  static_assert(!StringHasFromRangeAllocCtr<StringWithAlloc, InputRangeNotIndirectlyReadable, Alloc>);
  static_assert(!StringHasFromRangeAllocCtr<StringWithAlloc, InputRangeNotInputOrOutputIterator, Alloc>);
  // Not an allocator.
  static_assert(!StringHasFromRangeAllocCtr<StringWithAlloc, InputRange<char>, Empty>);

  return true;
}

template <class Iter, class Sent, class Alloc>
constexpr void test_with_input(std::vector<char> input) {
  { // (range)
    std::ranges::subrange in(Iter(input.data()), Sent(Iter(input.data() + input.size())));
    std::string c(std::from_range, in);

    LIBCPP_ASSERT(c.__invariants());
    assert(c.size() == static_cast<std::size_t>(std::distance(c.begin(), c.end())));
    assert(std::ranges::equal(input, c));
    LIBCPP_ASSERT(is_string_asan_correct(c));
  }

  { // (range, allocator)
    std::ranges::subrange in(Iter(input.data()), Sent(Iter(input.data() + input.size())));
    Alloc alloc;
    std::basic_string<char, std::char_traits<char>, Alloc> c(std::from_range, in, alloc);

    LIBCPP_ASSERT(c.__invariants());
    assert(c.get_allocator() == alloc);
    assert(c.size() == static_cast<std::size_t>(std::distance(c.begin(), c.end())));
    assert(std::ranges::equal(input, c));
    LIBCPP_ASSERT(is_string_asan_correct(c));
  }

  { // Ensure input-only sized ranges are accepted.
    using input_iter = cpp20_input_iterator<const char*>;
    const char in[]{'q', 'w', 'e', 'r'};
    std::string s(std::from_range, std::views::counted(input_iter{std::ranges::begin(in)}, std::ranges::ssize(in)));
    assert(s == "qwer");
  }
}

void test_string_exception_safety_throwing_allocator() {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  try {
    ThrowingAllocator<char> alloc;

    globalMemCounter.reset();
    // Note: the input string must be long enough to prevent SSO, otherwise the allocator won't be used.
    std::basic_string<char, std::char_traits<char>, ThrowingAllocator<char>> c(
        std::from_range, std::vector<char>(64, 'A'), alloc);
    assert(false); // The constructor call should throw.

  } catch (int) {
    assert(globalMemCounter.new_called == globalMemCounter.delete_called);
  }
#endif
}

constexpr bool test_inputs() {
  for_all_iterators_and_allocators<char>([]<class Iter, class Sent, class Alloc>() {
    // Shorter input -- SSO.
    test_with_input<Iter, Sent, Alloc>({'a', 'b', 'c', 'd', 'e'});
    // Longer input -- no SSO.
    test_with_input<Iter, Sent, Alloc>(std::vector<char>(64, 'A'));
    // Empty input.
    test_with_input<Iter, Sent, Alloc>({});
    // Single-element input.
    test_with_input<Iter, Sent, Alloc>({'a'});
  });

  return true;
}

#ifndef TEST_HAS_NO_LOCALIZATION
void test_counted_istream_view() {
  std::istringstream is{"qwert"};
  auto vals = std::views::istream<char>(is);
  std::string s(std::from_range, std::views::counted(vals.begin(), 3));
  assert(s == "qwe");
}
#endif

int main(int, char**) {
  test_inputs();
  static_assert(test_inputs());

  static_assert(test_constraints());

  // Note: `test_exception_safety_throwing_copy` doesn't apply because copying a `char` cannot throw.
  test_string_exception_safety_throwing_allocator();

#ifndef TEST_HAS_NO_LOCALIZATION
  test_counted_istream_view();
#endif

  return 0;
}
