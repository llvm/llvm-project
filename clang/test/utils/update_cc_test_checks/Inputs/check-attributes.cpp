// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
struct RT {
  char A;
  int B[10][20];
  char C;
};
struct ST {
  int X;
  double Y;
  struct RT Z;
};

int *foo(struct ST *s) {
  return &s[1].Z.B[5][13];
}
