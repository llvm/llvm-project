// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 %s -emit-llvm -fzos-extensions -triple s390x-none-zos -fvisibility=hidden -verify -o - | FileCheck %s

// Testing missing declarations.
#pragma export(d0) // expected-warning{{failed to resolve '#pragma export' to a declaration}}
#pragma export(f9) // expected-warning{{failed to resolve '#pragma export' to a declaration}}

// Testing pragma export after decl.
void f0(void) {}
static void sf0(void) {} // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 'sf0'}}
int v0;
static int s0; // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 's0'}}
#pragma export(f0)
#pragma export(sf0)
#pragma export(v0)
#pragma export(s0)

// Testing pragma export before decl.
#pragma export(f1)
#pragma export(sf1)
#pragma export(v1)
#pragma export(s1)
void f1(void) {}
static void sf1(void) {} // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 'sf1'}}
int v1;
static int s1; // expected-warning{{#pragma export is applicable to symbols with external linkage only; not applied to 's1'}}

void f2(void) {}

void t0(void) {}

// Testing pragma export after decl and usage.
#pragma export(f2)

// CHECK: @v0 = global i32
// CHECK: @v1 = global i32
// CHECK: define void @f0()
// CHECK: define void @f1()
// CHECK: define void @f2()
// CHECK: define hidden void @t0()
