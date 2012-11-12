// RUN: %clang  -emit-llvm -g -S %s -o - | FileCheck %s

// TAG_member is used to encode debug info for class constructor.
// CHECK: TAG_member
class A {
public:
  int z;
};

A *foo (A* x) {
  A *a = new A(*x);
  return a;
}

