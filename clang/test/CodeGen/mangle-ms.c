// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s

// CHECK: define void @"\01?f@@$$J0YAXP6AX@Z@Z"
__attribute__((overloadable)) void f(void (*x)()) {}

// CHECK: define void @f
void f(void (*x)(int)) {}

// CHECK: define void @g
void g(void (*x)(int)) {}

// CHECK: define void @"\01?g@@$$J0YAXP6AX@Z@Z"
__attribute__((overloadable)) void g(void (*x)()) {}
