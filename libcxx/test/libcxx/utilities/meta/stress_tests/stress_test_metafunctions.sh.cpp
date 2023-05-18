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
// _And:         3,498.639 ms     158 M
// __lazy_and:  10,138.982 ms     334 M
// __and_:      14,181.851 ms     648 M
//

// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %S/new.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace -std=c++17
// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %S/lazy.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace  -std=c++17 -DTEST_LAZY_AND
// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %S/std.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace   -std=c++17 -DTEST_STD_AND

#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "template_cost_testing.h"
using std::true_type;
using std::false_type;

#define FALSE_T() std::false_type,
#define TRUE_T() std::true_type,

#ifdef TEST_LAZY_AND
#define TEST_AND std::__lazy_and
#define TEST_OR std::__lazy_or
#elif defined(TEST_STD_AND)
#define TEST_AND std::__and_
#define TEST_OR std::__or_
#else
#define TEST_AND std::_And
#define TEST_OR std::_Or
#endif

void sink(...);

void Foo1(TEST_AND < REPEAT_1000(TRUE_T) true_type > t1) { sink(&t1); }
void Foo2(TEST_AND < REPEAT_1000(TRUE_T) REPEAT_1000(TRUE_T) true_type > t2) { sink(&t2); }
void Foo3(TEST_AND < REPEAT_1000(TRUE_T) true_type, false_type > t3) { sink(&t3); }
void Foo4(TEST_AND < REPEAT_1000(TRUE_T) REPEAT_1000(TRUE_T) true_type, false_type > t4) { sink(&t4); }
void Foo5(TEST_AND < false_type, REPEAT_1000(TRUE_T) true_type > t5) { sink(&t5); }
void Foo6(TEST_AND < false_type, REPEAT_1000(TRUE_T) REPEAT_1000(TRUE_T) true_type > t6) { sink(&t6); }

void escape() {

sink(&Foo1);
sink(&Foo2);
sink(&Foo3);
sink(&Foo4);
sink(&Foo5);
sink(&Foo6);
}
