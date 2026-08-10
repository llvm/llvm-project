// RUN: %clang_cc1 %s -verify
// RUN: %clang_cc1 %s -ast-dump | FileCheck %s
// expected-no-diagnostics

// PR42113: The following caused an assertion in mergeFunctionTypes
// because it causes one side to have an exception specification, which
// isn't typically supported in C.
void PR42113a();
void PR42113a(void) __attribute__((nothrow));
// CHECK: FunctionDecl {{.*}} PR42113a
// CHECK: FunctionDecl {{.*}} PR42113a
// CHECK: NoThrowAttr
void PR42113b() __attribute__((nothrow));
// CHECK: FunctionDecl {{.*}} PR42113b
// CHECK: NoThrowAttr
 __attribute__((nothrow)) void PR42113c();
// CHECK: FunctionDecl {{.*}} PR42113c
// CHECK: NoThrowAttr
