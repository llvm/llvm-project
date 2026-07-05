//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<charT> R>
//   constexpr basic_string& append_range(R&& rg); // C++23

#include <string>

#include "../../../../containers/sequences/insert_range_sequence_containers.h"
#include "test_macros.h"
#include "asan_testing.h"

// Tested cases:
// - different kinds of insertions (appending an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container);
// - an exception is thrown when allocating new elements.

constexpr bool test_constexpr() {
  for_all_iterators_and_allocators_constexpr<char, const char*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_append_range<std::basic_string<char, std::char_traits<char>, Alloc>, Iter, Sent>(
        []([[maybe_unused]] auto&& c) { LIBCPP_ASSERT(c.__invariants()); });
  });

  return true;
}

int main(int, char**) {
  static_assert(test_constraints_append_range<std::basic_string, char, int>());

  for_all_iterators_and_allocators<char, const char*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_append_range<std::basic_string<char, std::char_traits<char>, Alloc>, Iter, Sent>(
        []([[maybe_unused]] auto&& c) { LIBCPP_ASSERT(c.__invariants()); });
  });
  static_assert(test_constexpr());

  { // Check that `append_range` returns a reference to the string.
    std::string c;
    static_assert(std::is_lvalue_reference_v<decltype(c.append_range(FullContainer_Begin_EmptyRange<char>.input))>);
    assert(&c.append_range(FullContainer_Begin_EmptyRange<char>.input) == &c);
    LIBCPP_ASSERT(is_string_asan_correct(c));
    assert(&c.append_range(FullContainer_Begin_OneElementRange<char>.input) == &c);
    LIBCPP_ASSERT(is_string_asan_correct(c));
    assert(&c.append_range(FullContainer_Begin_MidRange<char>.input) == &c);
    LIBCPP_ASSERT(is_string_asan_correct(c));
    assert(&c.append_range(FullContainer_Begin_LongRange<char>.input) == &c);
    LIBCPP_ASSERT(is_string_asan_correct(c));
  }

  // Note: `test_append_range_exception_safety_throwing_copy` doesn't apply because copying chars cannot throw.
  {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    // Note: the input string must be long enough to prevent SSO, otherwise the allocator won't be used.
    std::string in(64, 'a');

    try {
      ThrowingAllocator<char> alloc;

      globalMemCounter.reset();
      std::basic_string<char, std::char_traits<char>, ThrowingAllocator<char>> c(alloc);
      c.append_range(in);
      assert(false); // The function call above should throw.

    } catch (int) {
      assert(globalMemCounter.new_called == globalMemCounter.delete_called);
    }
#endif
  }

  return 0;
}
