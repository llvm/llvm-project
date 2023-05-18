// RUN: %clang_cc1 -std=c++20 -triple=x86_64-linux-gnu -fmodules-codegen -emit-module-interface %s -o %t.pcm
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-linux-gnu %t.pcm -emit-llvm -o - | FileCheck %s

export module FooBar;

export {
  // CHECK-DAG: define{{.*}} i32 @_ZW6FooBar1fv(
  int f() { return 0; }
}

// CHECK-DAG: define weak_odr void @_ZW6FooBar2f2v(
inline void f2() {}

// CHECK-DAG: define{{.*}} void @_ZL2f3v(
static void f3() {}
export void use_f3() { f3(); }
