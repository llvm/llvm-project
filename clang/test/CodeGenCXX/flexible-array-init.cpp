// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-only -verify -DFAIL1 %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-only -verify -DFAIL2 %s

struct A { int x; int y[]; };
A a = { 1, 7, 11 };
// CHECK: @a ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 7, i32 11] }

A b = { 1, { 13, 15 } };
// CHECK: @b ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 13, i32 15] }

int f();
#ifdef FAIL1
A c = { f(), { f(), f() } }; // expected-error {{cannot compile this flexible array initializer yet}}
#endif
#ifdef FAIL2
void g() {
  static A d = { f(), { f(), f() } }; // expected-error {{cannot compile this flexible array initializer yet}}
}
#endif
