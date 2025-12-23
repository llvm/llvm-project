//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// enum class copy_options;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "check_bitmask_types.h"
#include "test_macros.h"
namespace fs = std::filesystem;

constexpr fs::copy_options ME(int val) { return static_cast<fs::copy_options>(val); }

// Verify binary operations on std::filesystem::copy_options bitmask constants.
// Also verify that compound assignment operators are not incorrectly marked [[nodiscard]],
// to avoid regression in https://llvm.org/PR171085.
constexpr bool test_bitmask_binary_operations() {
  using E = fs::copy_options;
  constexpr E bitmask_elems[]{
      // non-empty standard bitmask elements
      E::skip_existing,
      E::overwrite_existing,
      E::update_existing,
      E::recursive,
      E::copy_symlinks,
      E::skip_symlinks,
      E::directories_only,
      E::create_symlinks,
      E::create_hard_links,
  };

  for (auto elem : bitmask_elems) {
    assert((E::none | elem) == elem);
    assert((E::none & elem) == E::none);
    assert((E::none ^ elem) == elem);

    assert((elem | elem) == elem);
    assert((elem & elem) == elem);
    assert((elem ^ elem) == E::none);

    if (!TEST_IS_CONSTANT_EVALUATED) {
      {
        auto e = E::none;
        assert(&(e |= elem) == &e);
        assert(e == elem);
      }
      {
        auto e = E::none;
        assert(&(e &= elem) == &e);
        assert(e == E::none);
      }
      {
        auto e = E::none;
        assert(&(e ^= elem) == &e);
        assert(e == elem);
      }

      {
        auto e = elem;
        assert(&(e |= elem) == &e);
        assert(e == elem);
      }
      {
        auto e = elem;
        assert(&(e &= elem) == &e);
        assert(e == elem);
      }
      {
        auto e = elem;
        assert(&(e ^= elem) == &e);
        assert(e == E::none);
      }
    }
  }

  return true;
}

int main(int, char**) {
  typedef fs::copy_options E;
  static_assert(std::is_enum<E>::value, "");

  // Check that E is a scoped enum by checking for conversions.
  typedef std::underlying_type<E>::type UT;
  static_assert(!std::is_convertible<E, UT>::value, "");

  LIBCPP_STATIC_ASSERT(std::is_same<UT, unsigned short>::value, ""); // Implementation detail

  typedef check_bitmask_type<E, E::skip_existing, E::update_existing> BitmaskTester;
  assert(BitmaskTester::check());

  // The standard doesn't specify the numeric values of the enum.
  LIBCPP_STATIC_ASSERT(
          E::none == ME(0),
        "Expected enumeration values do not match");
  // Option group for copy_file
  LIBCPP_STATIC_ASSERT(
          E::skip_existing      == ME(1) &&
          E::overwrite_existing == ME(2) &&
          E::update_existing    == ME(4),
        "Expected enumeration values do not match");
  // Option group for copy on directories
  LIBCPP_STATIC_ASSERT(
          E::recursive == ME(8),
        "Expected enumeration values do not match");
  // Option group for copy on symlinks
  LIBCPP_STATIC_ASSERT(
          E::copy_symlinks == ME(16) &&
          E::skip_symlinks == ME(32),
        "Expected enumeration values do not match");
  // Option group for changing form of copy
  LIBCPP_STATIC_ASSERT(
          E::directories_only    == ME(64) &&
          E::create_symlinks     == ME(128) &&
          E::create_hard_links   == ME(256),
        "Expected enumeration values do not match");

  test_bitmask_binary_operations();
  static_assert(test_bitmask_binary_operations());

  return 0;
}
