// RUN: %clang_cc1 -E %s | FileCheck %s
// PR6282
// This test should not trigger the include guard optimization since
// the guard macro is defined on the first include.

#define ITERATING 1
#define X 1
#include "mi_opt2.h"
#undef X
#define X 2
#include "mi_opt2.h"

// CHECK: b: 1
// CHECK: b: 2

