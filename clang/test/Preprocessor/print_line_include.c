// RUN: %clang_cc1 -E -P %s | FileCheck %s
// CHECK: int x;
// CHECK-NEXT: int x;

#include "print_line_include.h"
#include "print_line_include.h"
