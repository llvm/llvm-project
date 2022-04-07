// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

// CHECK: getelementptr inbounds i32, i32* %{{vla|[0-9]}}
extern void f(int *);
int e(int m, int n) {
  int x[n];
  f(x);
  return x[m];
}
