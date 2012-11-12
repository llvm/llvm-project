// RUN: %clang  -emit-llvm -g -S %s -o - | FileCheck %s
// XFAIL: *

class B {
public:
  int bb;
  void fn2() {}
};

class A {
public:
  int aa;
  void fn1(B b) { b.fn2(); }
};

void foo(A *aptr) {
}

void bar() {
  A a;
}

// B should only be emitted as a forward reference (i32 4).
// CHECK: metadata !"B", metadata !6, i32 3, i32 0, i32 0, i32 0, i32 4} ; [ DW_TAG_class_type ]
