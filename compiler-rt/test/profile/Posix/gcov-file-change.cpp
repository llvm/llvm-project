// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: %clangxx --coverage main.cpp -o t
// RUN: %run ./t
// RUN: llvm-cov gcov -t t-main. | FileCheck %s

//--- main.cpp
#include "a.h"
#include <stdio.h>

// CHECK:      Runs:1
/// __cxx_global_var_init contains a block from a.h. Don't attribute its lines to main.cpp.
// CHECK-NOT:  {{^ +[0-9]+:}}

inline auto *const inl_var_main = // CHECK:      1: [[#]]:inline auto
    new A;                        // CHECK-NEXT: 1: [[#]]:
void foo(int x) {                 // CHECK-NEXT: 1: [[#]]:
  if (x) {                        // CHECK-NEXT: 1: [[#]]:
#include "a.inc"
  }
}
// CHECK-NOT:  {{^ +[0-9]+:}}

int main(int argc, char *argv[]) { // CHECK:      1: [[#]]:int main
  foo(1);                          // CHECK-NEXT: 1: [[#]]:
}                                  // CHECK-NEXT: 1: [[#]]:
// CHECK-NOT:  {{^ +[0-9]+:}}

// CHECK:      Source:a.h
// CHECK:      1: 1:struct A
// CHECK-NOT:  {{^ +[0-9]+:}}

//--- a.h
/// Apple targets doesn't enable -mconstructor-aliases by default and the count may be 4.
struct A { A() { } };              // CHECK:      {{[24]}}: [[#]]:struct A
inline auto *const inl_var_a =
    new A;
/// TODO a.inc:1 should have line execution.
// CHECK-NOT:  {{^ +[0-9]+:}}

//--- a.inc
puts("");
