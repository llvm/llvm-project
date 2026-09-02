// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -disable-O0-optnone -triple x86_64-pc-linux-gnu -O0 -fmerge-functions -emit-llvm -o - -x c++ < %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O1 -fmerge-functions -emit-llvm -o - -x c++ < %s | FileCheck %s

// Basic functionality test. Function merging doesn't kick in on functions that
// are too simple.

// Apply -disable-O0-optnone at -O0 to test that function merging can still
// occur at -O0 if it's not disable via optnone/noipa (this can happen in other
// cases, like with the minsize attribute which also disables optnone behavior)

struct A {
  virtual int f(int x, int *p) { return x ? *p : 1; }
  virtual int g(int x, int *p) { return x ? *p : 1; }
} a;

// CHECK: define linkonce_odr noundef i32 @_ZN1A1gEiPi(
// CHECK:   tail call noundef i32 @0(

// CHECK: define linkonce_odr noundef i32 @_ZN1A1fEiPi(
// CHECK:   tail call noundef i32 @0(
