// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct A {
  A() { x = 10; }
  int x;
};

const A x;

// CHECK: @_ZL1x = internal global

struct X {
  int (*fp)(int, int);
};

int add(int x, int y) { return x + y; }

// CHECK: @x2 = constant
extern const X x2;
const X x2 = { &add };

struct X1 {
  mutable int i;
};

struct X2 {
  X1 array[3];
};

// CHECK: @x2b = global
extern const X2 x2b;
const X2 x2b = { { { 1 }, { 2 }, { 3 } } };
