// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// The following compilers don't consider a type an aggregate type (and
// consequently not a literal type) if it has a base class at all.
// In C++17, an aggregate type is allowed to have a base class if it's not
// virtual, private, nor protected (e.g. ConstexprTestTypes:::NoCtors).
// XFAIL: gcc-5, gcc-6
// XFAIL: clang-3.5, clang-3.6, clang-3.7, clang-3.8
// XFAIL: apple-clang-6, apple-clang-7, apple-clang-8.0

// <variant>

// template <class ...Types> class variant;

// constexpr bool valueless_by_exception() const noexcept;

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "archetypes.hpp"
#include "test_macros.h"
#include "variant_test_helpers.hpp"

int main() {
#if TEST_STD_VER == 17
  { // This test does not pass on C++20 or later; see https://bugs.llvm.org/show_bug.cgi?id=39232
    using V = std::variant<int, ConstexprTestTypes::NoCtors>;
    constexpr V v;
    static_assert(!v.valueless_by_exception(), "");
  }
#endif
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
}
