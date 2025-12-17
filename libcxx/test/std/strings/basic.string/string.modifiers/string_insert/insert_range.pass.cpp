//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<charT> R>
//   constexpr iterator insert_range(const_iterator p, R&& rg);                                // C++23

#include <sstream>
#include <string>

#include "../../../../containers/sequences/insert_range_sequence_containers.h"
#include "test_macros.h"

// Tested cases:
// - different kinds of insertions (inserting an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container at the {beginning/middle/end});
// - an exception is thrown when allocating new elements.

constexpr bool test_constexpr() {
  for_all_iterators_and_allocators_constexpr<char, const char*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_insert_range<std::basic_string<char, std::char_traits<char>, Alloc>, Iter, Sent>(
        []([[maybe_unused]] auto&& c) { LIBCPP_ASSERT(c.__invariants()); });
  });

  { // Ensure input-only sized ranges are accepted.
    using input_iter = cpp20_input_iterator<const char*>;
    const char in[]{'q', 'w', 'e', 'r'};
    std::string s = "zxcv";
    s.insert_range(s.begin(), std::views::counted(input_iter{std::ranges::begin(in)}, std::ranges::ssize(in)));
    assert(s == "qwerzxcv");
  }

  return true;
}

#ifndef TEST_HAS_NO_LOCALIZATION
void test_counted_istream_view() {
  std::istringstream is{"qwert"};
  auto vals     = std::views::istream<char>(is);
  std::string s = "zxcv";
  s.insert_range(s.begin(), std::views::counted(vals.begin(), 3));
  assert(s == "qwezxcv");
}
#endif

int main(int, char**) {
  static_assert(test_constraints_insert_range<std::basic_string, char, int>());

  for_all_iterators_and_allocators<char, const char*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_insert_range<std::basic_string<char, std::char_traits<char>, Alloc>, Iter, Sent>(
        []([[maybe_unused]] auto&& c) { LIBCPP_ASSERT(c.__invariants()); });
  });
  static_assert(test_constexpr());

#ifndef TEST_HAS_NO_LOCALIZATION
  test_counted_istream_view();
#endif

  // Note: `test_insert_range_exception_safety_throwing_copy` doesn't apply because copying chars cannot throw.
  {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    // Note: the input string must be long enough to prevent SSO, otherwise the allocator won't be used.
    std::string in(64, 'a');

    try {
      ThrowingAllocator<char> alloc;

      globalMemCounter.reset();
      std::basic_string<char, std::char_traits<char>, ThrowingAllocator<char>> c(alloc);
      c.insert_range(c.end(), in);
      assert(false); // The function call above should throw.

    } catch (int) {
      assert(globalMemCounter.new_called == globalMemCounter.delete_called);
    }
#endif
  }

  return 0;
}
