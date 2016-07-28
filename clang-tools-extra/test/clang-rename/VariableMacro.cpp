// RUN: clang-rename -offset=147 -new-name=Z %s -- | FileCheck %s

#define Y X // CHECK: #define Y Z

void foo(int value) {}

void macro() {
  int X;    // CHECK: int Z;
  X = 42;   // CHECK: Z = 42;
  Y -= 0;
  foo(X);   // CHECK: foo(Z);
  foo(Y);
}

// Use grep -FUbo 'X' <file> to get the correct offset of X when changing
// this file.
