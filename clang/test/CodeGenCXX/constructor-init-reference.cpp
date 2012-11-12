// RUN: %clang_cc1 -emit-llvm -o - %s | grep "store i32\* @x, i32\*\*"

int x;
struct A {
  int& y;
  A() : y(x) {}
};
A z;

