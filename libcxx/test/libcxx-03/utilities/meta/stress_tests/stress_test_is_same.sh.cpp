//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a dummy feature that prevents this test from running by default.
// REQUIRES: template-cost-testing

// The table below compares the compile time and object size for each of the
// variants listed in the RUN script.
//
//  Impl          Compile Time    Object Size
// -------------------------------------------
// std::_IsSame:    689.634 ms     356 K
// std::is_same:  8,129.180 ms     560 K
//
// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %S/orig.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace -std=c++17
// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %S/new.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace -std=c++17 -DTEST_NEW

#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "template_cost_testing.h"

template <int N> struct Arg { enum { value = 1 }; };

#ifdef TEST_NEW
#define IS_SAME  std::_IsSame
#else
#define IS_SAME std::is_same
#endif

#define TEST_CASE_NOP() IS_SAME < Arg< __COUNTER__ >, Arg < __COUNTER__ > >::value,
#define TEST_CASE_TYPE() IS_SAME < Arg< __COUNTER__ >, Arg < __COUNTER__ > >,

int sink(...);

int x = sink(
  REPEAT_10000(TEST_CASE_NOP)
  REPEAT_10000(TEST_CASE_NOP) 42
);

void Foo( REPEAT_1000(TEST_CASE_TYPE) int) { }

static_assert(__COUNTER__ > 10000, "");

void escape() {

sink(&x);
sink(&Foo);
}
