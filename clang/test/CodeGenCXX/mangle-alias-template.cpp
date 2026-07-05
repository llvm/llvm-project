// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

template<typename T> struct alloc {};
template<typename T> using Alloc = alloc<T>;
template<typename T, typename A = Alloc<T>> struct vector {};

template<typename T> using Vec = vector<T>;

template<typename T> void f(Vec<T> v);
template<typename T> void g(T);

template<template<typename> class F> void h(F<int>);

// CHECK-LABEL: define{{.*}} void @_Z1zv(
void z() {
  vector<int> VI;
  f(VI);
  // CHECK: call void @_Z1fIiEv6vectorIT_5allocIS1_EE(

  Vec<double> VD;
  g(VD);
  // CHECK: call void @_Z1gI6vectorId5allocIdEEEvT_(

  h<Vec>(VI);
  // CHECK: call void @_Z1hI3VecEvT_IiE(

  Alloc<int> AC;
  h(AC);
  // CHECK: call void @_Z1hI5allocEvT_IiE(

  h<Alloc>(AC);
  // CHECK: call void @_Z1hI5AllocEvT_IiE(

  Vec<char> VC;
  g<Vec<char>>(VC);
  // CHECK: call void @_Z1gI6vectorIc5allocIcEEEvT_(

  Vec<Vec<int>> VVI;
  g(VVI);
  // CHECK: call void @_Z1gI6vectorIS0_Ii5allocIiEES1_IS3_EEEvT_(
}
