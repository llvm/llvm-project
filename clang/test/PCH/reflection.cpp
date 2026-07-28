// Test without PCH
// RUN: %clang_cc1 -std=c++26 -freflection %s -include %S/reflection_include.h -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++26 -freflection -x c++-header %S/reflection_include.h -emit-pch -o %t
// RUN: %clang_cc1 -std=c++26 -freflection -include-pch %t -verify %s

// expected-no-diagnostics


static_assert(^^int == ^^int);
static_assert(^^int != ^^double);
static_assert(info{} != ^^int);
static_assert(__is_same(decltype(^^int), info));
static_assert(__is_same(decltype(^^double), info));
