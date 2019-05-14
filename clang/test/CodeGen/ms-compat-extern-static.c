// RUN: %clang_cc1 -emit-llvm %s -o - -fms-extensions -triple x86_64-windows | FileCheck %s

// CHECK: @n = internal global i32 1
extern int n;
static int n = 1;
int *use = &n;

// CHECK: define internal void @f(
extern void f();
static void f() {}
void g() { return f(); }
