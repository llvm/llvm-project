// RUN: %clang_cc1 -fmodules-ts -std=c++1z -triple=x86_64-linux-gnu -fmodules-codegen -emit-module-interface %s -o %t.pcm
// RUN: %clang_cc1 -fmodules-ts -std=c++1z -triple=x86_64-linux-gnu %t.pcm -emit-llvm -o - | FileCheck %s

export module FooBar;

export {
  // CHECK-DAG: define i32 @_Z1fv(
  int f() { return 0; }
}

// CHECK-DAG: define weak_odr void @_ZW6FooBarE2f2v(
inline void f2() { }

// CHECK-DAG: define void @_ZW6FooBarE2f3v(
static void f3() {}
export void use_f3() { f3(); }

// FIXME: Emit global variables and their initializers with this TU.
// Emit an initialization function that other TUs can call, with guard variable?

// FIXME: const-qualified variables don't have implicit internal linkage when owned by a module.
