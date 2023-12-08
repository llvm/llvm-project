// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s

__attribute__((section("A")))
const int a = 1;
const int *f() { return &a; }
// CHECK: @_ZL1a = internal constant i32 1, section "A"

int init();
__attribute__((section("B")))
const int b = init();
// Even if it's const-qualified, it must not be LLVM IR `constant` since it's
// dynamically initialised.
// CHECK: @_ZL1b = internal global i32 0, section "B"

__attribute__((section("C")))
int c = 2;
// CHECK: @c = {{.*}}global i32 2, section "C"

__attribute__((section("D")))
int d = init();
// CHECK: @d = {{.*}}global i32 0, section "D"

__attribute__((section("E")))
int e;
// CHECK: @e = {{.*}}global i32 0, section "E", align 4
