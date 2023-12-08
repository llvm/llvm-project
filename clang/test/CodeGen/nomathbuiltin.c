// RUN: %clang_cc1 -fno-math-builtin -emit-llvm -o - %s | FileCheck %s

// Check that the -fno-math-builtin option for -cc1 is working properly.


double pow(double, double);

double foo(double a, double b) {
  return pow(a, b);
// CHECK: call {{.*}}double @pow
}

