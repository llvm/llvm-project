// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

template<bool b> int f() noexcept(b) {}
decltype(f<false>()) a;
decltype(f<true>()) b;

#else

static_assert(!noexcept(f<false>()), "");
static_assert(noexcept(f<true>()), "");

#endif
