// RUN: %clang_cc1 -std=c++1z -emit-pch -o %t.1.ast %S/Inputs/function-1.cpp
// RUN: %clang_cc1 -std=c++1z  -ast-merge %t.1.ast -fsyntax-only %s 2>&1 | FileCheck %s
// XFAIL: *

static_assert(add(1, 2) == 5);

// FIXME: support of templated function overload is still not implemented.
static_assert(add('\1', '\2') == 3);

// CHECK-NOT: static_assert
