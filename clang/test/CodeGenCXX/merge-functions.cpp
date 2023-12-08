// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O0 -fmerge-functions -emit-llvm -o - -x c++ < %s | FileCheck %s -implicit-check-not=_ZN1A1gEiPi
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O1 -fmerge-functions -emit-llvm -o - -x c++ < %s | FileCheck %s -implicit-check-not=_ZN1A1gEiPi

// Basic functionality test. Function merging doesn't kick in on functions that
// are too simple.

struct A {
  virtual int f(int x, int *p) { return x ? *p : 1; }
  virtual int g(int x, int *p) { return x ? *p : 1; }
} a;

// CHECK: define {{.*}} @_ZN1A1fEiPi
