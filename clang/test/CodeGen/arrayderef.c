// RUN: %clang_cc1 %s -emit-llvm -O1 -o - | FileCheck %s
// The load here was getting lost because this code was close
// enough to the traditional (wrong) implementation of offsetof
// to confuse the gcc FE.  8629268.

struct foo {
  int x;
  int *y;
};

struct foo Foo[1];

int * bar(unsigned int ix) {
// CHECK: load
  return &Foo->y[ix];
}
