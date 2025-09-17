// RUN: %clang_cc1 -foverflow-behavior-types -std=c++11 -ast-print %s -o - | FileCheck %s

extern int __attribute__((overflow_behavior(no_wrap))) a;
extern int __attribute__((overflow_behavior(wrap))) b;

// CHECK: extern __no_wrap int a;
// CHECK: extern __wrap int b;
