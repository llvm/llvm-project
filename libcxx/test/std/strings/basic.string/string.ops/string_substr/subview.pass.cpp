//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <string>

//    constexpr basic_string_view<charT, traits> subview(size_type pos = 0,
//                                                       size_type n = npos) const;

#include <cassert>
#include <concepts>
#include <string>
#include <string_view>
#include <utility>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"
#include "type_algorithms.h"

#define CS(S) MAKE_CSTRING(CharT, S)

template <typename CharT, typename TraitsT, typename AllocT>
constexpr void test() {
  std::basic_string<CharT, TraitsT, AllocT> s{CS("Hello cruel world!"), AllocT{}};

  { // With a default position and a character length.

    // Also check if subview() is a const-qualified.
    assert(std::as_const(s).subview() == CS("Hello cruel world!"));

    // Check it the return type of subview() is correct.
    std::same_as<std::basic_string_view<CharT, TraitsT>> decltype(auto) sv = s.subview();
    assert(sv == CS("Hello cruel world!"));
  }

  { // Check with different position and length.

    // With a explict position and a character length.
    assert(s.subview(6, 5) == CS("cruel"));

    // From the beginning of the string with a explicit character length.
    assert(s.subview(0, 5) == CS("Hello"));

    // To the end of string with the default character length.
    assert(s.subview(12) == CS("world!"));

    // From the beginning to the end of the string with explicit values.
    assert(s.subview(0, s.size()) == CS("Hello cruel world!"));
  }

  // Test if exceptions are thrown correctly.
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!std::is_constant_evaluated()) {
    { // With a position that is out of range.
      try {
        std::ignore = s.subview(s.size() + 1);
        assert(false);
      } catch ([[maybe_unused]] const std::out_of_range& ex) {
        LIBCPP_ASSERT(std::string(ex.what()) == "string_view::substr");
      } catch (...) {
        assert(false);
      }
    }

    { // With a position that is out of range and a 0 character length.
      try {
        std::ignore = s.subview(s.size() + 1, 0);
        assert(false);
      } catch ([[maybe_unused]] const std::out_of_range& ex) {
        LIBCPP_ASSERT(std::string(ex.what()) == "string_view::substr");
      } catch (...) {
        assert(false);
      }
    }

    { // With a position that is out of range and a some character length.
      try {
        std::ignore = s.subview(s.size() + 1, 1);
        assert(false);
      } catch ([[maybe_unused]] const std::out_of_range& ex) {
        LIBCPP_ASSERT(std::string(ex.what()) == "string_view::substr");
      } catch (...) {
        assert(false);
      }
    }
  }
#endif
}

template <typename CharT>
constexpr void test() {
  test<CharT, std::char_traits<CharT>, std::allocator<CharT>>();
  test<CharT, std::char_traits<CharT>, min_allocator<CharT>>();
  test<CharT, std::char_traits<CharT>, safe_allocator<CharT>>();
  test<CharT, std::char_traits<CharT>, test_allocator<CharT>>();

  test<CharT, constexpr_char_traits<CharT>, std::allocator<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, min_allocator<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, safe_allocator<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>();
}

constexpr bool test() {
  types::for_each(types::character_types(), []<typename CharT> { test<CharT>(); });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
