// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-only -verify -DFAIL1 %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-only -verify -DFAIL2 %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-only -verify -DFAIL3 %s

struct A { int x; int y[]; };
A a = { 1, 7, 11 };
// CHECK: @a ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 7, i32 11] }

A b = { 1, { 13, 15 } };
// CHECK: @b ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 13, i32 15] }

char f();
#ifdef FAIL1
A c = { f(), { f(), f() } }; // expected-error {{cannot compile this flexible array initializer yet}}
#endif
#ifdef FAIL2
void g() {
  static A d = { f(), { f(), f() } }; // expected-error {{cannot compile this flexible array initializer yet}}
}
#endif
#ifdef FAIL3
struct B { int x; char y; char z[]; };
B e = {f(), f(), f(), f()}; // expected-error {{cannot compile this flexible array initializer yet}}
#endif

namespace zero_initializer {
A a0{0, 0}, a1{0, {0, 0}};
// CHECK: @_ZN16zero_initializer2a0E = global { i32, [1 x i32] } zeroinitializer,
// CHECK: @_ZN16zero_initializer2a1E = global { i32, [2 x i32] } zeroinitializer,

struct S { int x[]; };

S s{0};
// CHECK: @_ZN16zero_initializer1sE = global { [1 x i32] } zeroinitializer,
}
