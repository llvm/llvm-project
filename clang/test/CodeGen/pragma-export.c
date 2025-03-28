// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 %s -emit-llvm -fzos-extensions -triple s390x-none-zos -fvisibility=hidden -verify -o - | FileCheck %s

// Testing pragma export after decl.
void f0(void) {}
int v0;
#pragma export(f0)
#pragma export(v0)

// Testing pragma export before decl.
#pragma export(f1)
#pragma export(v1)
void f1(void) {}
int v1;

void f2(void) {}

void t0(void) { f2();}

// Testing pragma export after decl and usage.
#pragma export(f2)

// CHECK: @v0 = global i32
// CHECK: @v1 = global i32
// CHECK: define void @f0()
// CHECK: define void @f1()
// CHECK: define void @f2()
// CHECK: define hidden void @t0()
