// RUN: %clang_cc1 -E -P %s | FileCheck %s
// CHECK: int x;
// CHECK-NEXT: int x;

// RUN: %clang_cc1 -E -P -fminimize-whitespace %s | FileCheck %s --check-prefix=MINWS --strict-whitespace
// MINWS: {{^}}int x;int x;{{$}}

#include "print_line_include.h"
#include "print_line_include.h"
