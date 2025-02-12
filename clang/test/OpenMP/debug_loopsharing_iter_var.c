// This testcase checks emission of debug info for iterator variables of
// worksharing loop.

// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -debug-info-kind=constructor -x c -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// CHECK-LABEL: define internal void @main.omp_outlined_debug__
// CHECK: %i = alloca i32, align 4
// CHECK: #dbg_declare(ptr %i, [[IVAR:![0-9]+]], !DIExpression()
// CHECK-NOT: %i4 = alloca i32, align 4
// CHECK-NOT: #dbg_declare(ptr %i4, [[IVAR]]]

extern int printf(const char *, ...);

int main(int argc, char **argv) {
  const int n = 100 * argc;
  double a[n], total=42., c = .3;
#pragma omp parallel for reduction(+ : total)
  for (int i = 0; i < n; i++) {
    total += a[i] = i * c;
  }
  printf("total=%lf, expected:%lf, a[50]=%lf\n", total, c * n * (n - 1) / 2, a[50]);
}
