// RUN: %clang_cc1 -fms-extensions %s -verify
// RUN: %clang_cc1 -E -fms-extensions %s | FileCheck %s
// expected-no-diagnostics

#define CAT(a,b) a##b

char foo\u00b5;
char*p = &CAT(foo, \u00b5);
// CHECK: char fooµ;
// CHECK-NEXT: char*p = &fooµ;
