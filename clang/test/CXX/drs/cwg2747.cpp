// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=expected %s -E | FileCheck %s --strict-whitespace --allow-empty

// expected-no-diagnostics
// cwg2747: yes

// Check that a newline is still added even though there is a
// physical newline at the end of the file (which is spliced)
// CHECK: int x;{{$[[:space:]]^}}int y;int z;{{$[[:space:]]^$}}
int x;
int y;\
int z;\
