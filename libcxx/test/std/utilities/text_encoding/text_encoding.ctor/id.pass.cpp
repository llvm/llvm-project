//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26

// class text_encoding

// text_encoding::text_encoding(id) noexcept

// Concerns:
// 1. text_encoding(id) must be nothrow
// 2. Constructing an object with a valid id must set mib() and the name to the corresponding value.
// 3. Constructing an object using id::unknown must set mib() to id::unknown and the name to an empty string.
// 4. Constructing an object using id::other must set mib() to id::other and the name to an empty string.

#include "test_text_encoding.h"

using te_id = std::text_encoding::id;

constexpr bool id_ctor(te_id i, te_id expect_id, std::string_view expect_name) {
  auto te = std::text_encoding(i);
  if (te.mib() != expect_id) {
    return false;
  }
  if (expect_name.compare(te.name()) != 0) {
    return false;
  }
  if (!std::ranges::contains(te.aliases(), std::string_view(te.name()))) {
    return false;
  }
  return true;
}

constexpr bool id_ctors() {
  for (auto pair : unique_encoding_data) {
    if (!id_ctor(te_id{pair.mib}, te_id{pair.mib}, pair.name)) {
      return false;
    }
  }
  return true;
}

constexpr bool test_unknown() {
  constexpr auto te = std::text_encoding(te_id::unknown);
  if (te.mib() != te_id::unknown) {
    return false;
  }
  if (std::string_view("").compare(te.name()) != 0) {
    return false;
  }
  if (!std::ranges::empty(te.aliases())) {
    return false;
  }
  return true;
}

constexpr bool test_other() {
  constexpr auto te = std::text_encoding(te_id::other);
  if (te.mib() != te_id::other) {
    return false;
  }
  if (std::string_view("").compare(te.name()) != 0) {
    return false;
  }
  if (!std::ranges::empty(te.aliases())) {
    return false;
  }
  return true;
}

int main() {
  {
    static_assert(std::is_nothrow_constructible<std::text_encoding, std::text_encoding::id>::value,
                  "Must be nothrow constructible with id");
  }

  {
    static_assert(id_ctors());
    assert(id_ctors());
  }

  {
    static_assert(test_unknown());
    assert(test_unknown());
  }

  {
    static_assert(test_other());
    assert(test_other());
  }
}
