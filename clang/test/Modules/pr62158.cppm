// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/lib.cppm -o %t/lib.pcm
// RUN: %clang_cc1 -std=c++20 %t/main.cpp -fmodule-file=lib=%t/lib.pcm \
// RUN:     -verify -fsyntax-only

//--- header.h
namespace lib::inline __1 {
template <class>
inline constexpr bool test = false;
template <class>
constexpr bool func() {
    return false;
}
inline constexpr bool non_templ = true;
} // namespace lib

//--- lib.cppm
module;
#include "header.h"
export module lib;

export namespace lib {
    using lib::test;
    using lib::func;
    using lib::non_templ;
} // namespace lib

//--- main.cpp
// expected-no-diagnostics
import lib;

struct foo {};

template <>
inline constexpr bool lib::test<foo> = true;

template <>
constexpr bool lib::func<foo>() {
    return true;
}

static_assert(lib::test<foo>);
static_assert(lib::func<foo>());
