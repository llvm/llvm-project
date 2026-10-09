//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Make sure insert catches a position that isn't in [begin(), end()], such as an iterator into
// another string or an iterator invalidated by a reallocation.

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// std::string::insert(const_iterator, char) is instantiated in the dylib, so we need an up-to-date one
// XFAIL: using-built-library-before-llvm-24

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 23
#  include <string_view>
#endif

template <class String>
void test_invalid_position(String& s, typename String::const_iterator pos) {
  [[maybe_unused]] const char* message = "string::insert called with an iterator outside [begin(), end()]";
  const char a[]                       = "abc";
  TEST_LIBCPP_ASSERT_FAILURE(s.insert(pos, 'x'), message);
  TEST_LIBCPP_ASSERT_FAILURE(s.insert(pos, 2, 'x'), message);
  TEST_LIBCPP_ASSERT_FAILURE(s.insert(pos, a, a + 3), message);
  TEST_LIBCPP_ASSERT_FAILURE(
      s.insert(pos, forward_iterator<const char*>(a), forward_iterator<const char*>(a + 3)), message);
  TEST_LIBCPP_ASSERT_FAILURE(
      s.insert(pos, cpp17_input_iterator<const char*>(a), cpp17_input_iterator<const char*>(a + 3)), message);
  TEST_LIBCPP_ASSERT_FAILURE((s.insert(pos, {'a', 'b', 'c'})), message);
#if TEST_STD_VER >= 23
  TEST_LIBCPP_ASSERT_FAILURE(s.insert_range(pos, std::string_view("abc")), message);
#endif
}

template <class String>
void test() {
  // An iterator into another string, for short and long strings.
  {
    String s("123");
    String other("456");
    test_invalid_position(s, other.begin());
    test_invalid_position(s, other.end());
  }
  {
    String s("a string that is too long for the small buffer");
    String other("another string that is too long for the small buffer");
    test_invalid_position(s, other.begin());
    test_invalid_position(s, other.end());
  }

  // An iterator invalidated by a reallocation.
  {
    String s("a string that is too long for the small buffer");
    typename String::const_iterator stale = s.begin();
    s.reserve(s.capacity() + 1);
    test_invalid_position(s, stale);
  }

  // An iterator past end() that is still inside the allocation.
  {
    String s("123");
    typename String::const_iterator stale = s.end();
    s.pop_back();
    test_invalid_position(s, stale);
  }
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return 0;
}
