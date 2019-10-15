// RUN: %clang -O1 -fexperimental-new-pass-manager -fno-unroll-loops -S -o - %s -emit-llvm | FileCheck %s
// RUN: %clang -O1 -fno-experimental-new-pass-manager -fno-unroll-loops -S -o - %s -emit-llvm | FileCheck %s
extern int a[16];
int b = 0;
int foo(void) {
#pragma unroll
  for (int i = 0; i < 16; ++i)
    a[i] = b += 2;
  return b;
}
// CHECK-NOT: br i1

