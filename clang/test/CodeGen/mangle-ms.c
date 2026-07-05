// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s

// CHECK: define dso_local void @"?f@@$$J0YAXP6AX@Z@Z"
__attribute__((overloadable)) void f(void (*x)()) {}

// CHECK: define dso_local void @f
void f(void (*x)(int)) {}

// CHECK: define dso_local void @g
void g(void (*x)(int)) {}

// CHECK: define dso_local void @"?g@@$$J0YAXP6AX@Z@Z"
__attribute__((overloadable)) void g(void (*x)()) {}
