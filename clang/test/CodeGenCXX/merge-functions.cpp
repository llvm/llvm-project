// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O0 -fmerge-functions -emit-llvm -o - -x c++ < %s | FileCheck %s --check-prefix=O0 --implicit-check-not=call
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-pc-linux-gnu -O0 -fmerge-functions -emit-llvm -o - -x c++ < %s | FileCheck %s --check-prefix=O0 --implicit-check-not=call %}
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O1 -fmerge-functions -emit-llvm -o - -x c++ < %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-pc-linux-gnu -O1 -fmerge-functions -emit-llvm -o - -x c++ < %s | FileCheck %s %}

// Basic functionality test. Function merging doesn't kick in on functions that
// are too simple.

struct A {
  virtual int f(int x, int *p) { return x ? *p : 1; }
  virtual int g(int x, int *p) { return x ? *p : 1; }
} a;

// CHECK: define linkonce_odr noundef i32 @_ZN1A1gEiPi(
// CHECK:   tail call noundef i32 @0(

// CHECK: define linkonce_odr noundef i32 @_ZN1A1fEiPi(
// CHECK:   tail call noundef i32 @0(

// At -O0 every function is optnone, which implies noipa, so they're not merged.
// O0: define linkonce_odr noundef i32 @_ZN1A1fEiPi(
// O0: define linkonce_odr noundef i32 @_ZN1A1gEiPi(
