// RUN: not %clang_cc1 -g -emit-llvm %s

// Don't attempt to codegen invalid code that would lead to a crash

// PR16933
struct A;
A *x;
struct A {
  B y;
};
A y;
