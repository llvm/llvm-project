// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-pc -emit-llvm -o - %s | FileCheck %s

template<class T, class U> struct A {};

template<template<class> class TT> void f(TT<int>);

// CHECK-LABEL: define{{.*}} void @_Z1zv(
void z() {
  f(A<int, double>());
  // CHECK: call void @_Z1fITtTyE1A1IdEEvT_IiE()
}
