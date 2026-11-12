// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s

template <typename>
struct __declspec(dllimport) S {
  S();
};

template <typename T>
struct __declspec(dllimport) U {
  static S<T> u;
};

template <typename T>
S<T> U<T>::u;

template S<int> U<int>::u;
// CHECK-NOT: define internal void @"??__Eu@?$U@H@@2U?$S@H@@A@YAXXZ"(

S<int> &i = U<int>::u;
