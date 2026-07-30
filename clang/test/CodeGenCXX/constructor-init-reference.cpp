// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

int x;
struct A {
  int& y;
  A() : y(x) {}
};
A z;
// CHECK: store ptr @x, ptr
