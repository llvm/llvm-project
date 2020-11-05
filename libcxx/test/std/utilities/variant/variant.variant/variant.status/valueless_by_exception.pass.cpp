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

// constexpr bool valueless_by_exception() const noexcept;

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
    static_assert(!v.valueless_by_exception(), "");
  }
  {
    using V = std::variant<int, long>;
    V v;
    assert(!v.valueless_by_exception());
  }
  {
    using V = std::variant<int, long, std::string>;
    const V v("abc");
    assert(!v.valueless_by_exception());
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using V = std::variant<int, MakeEmptyT>;
    V v;
    assert(!v.valueless_by_exception());
    makeEmpty(v);
    assert(v.valueless_by_exception());
  }
#endif

  return 0;
}
