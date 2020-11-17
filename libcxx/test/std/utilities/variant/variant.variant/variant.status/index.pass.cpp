// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_variant_access is supported starting in macosx10.13
// XFAIL: with_system_cxx_lib=macosx10.12 && !no-exceptions
// XFAIL: with_system_cxx_lib=macosx10.11 && !no-exceptions
// XFAIL: with_system_cxx_lib=macosx10.10 && !no-exceptions
// XFAIL: with_system_cxx_lib=macosx10.9 && !no-exceptions

// <variant>

// template <class ...Types> class variant;

// constexpr size_t index() const noexcept;

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "archetypes.h"
#include "test_macros.h"
#include "variant_test_helpers.h"


int main(int, char**) {
  {
    using V = std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
  }
  {
    using V = std::variant<int, long>;
    V v;
    assert(v.index() == 0);
  }
  {
    using V = std::variant<int, long>;
    constexpr V v(std::in_place_index<1>);
    static_assert(v.index() == 1, "");
  }
  {
    using V = std::variant<int, std::string>;
    V v("abc");
    assert(v.index() == 1);
    v = 42;
    assert(v.index() == 0);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using V = std::variant<int, MakeEmptyT>;
    V v;
    assert(v.index() == 0);
    makeEmpty(v);
    assert(v.index() == std::variant_npos);
  }
#endif

  return 0;
}
