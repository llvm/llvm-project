// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=208 -new-name=Z %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

#define Y X // CHECK: #define Y Z

void foo(int value) {}

void macro() {
  int X;    // CHECK: int Z;
  X = 42;   // CHECK: Z = 42;
  Y -= 0;
  foo(X);   // CHECK: foo(Z);
  foo(Y);
}

// Use grep -FUbo 'foo;' <file> to get the correct offset of foo when changing
// this file.
