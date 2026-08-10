// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s

// Define __complex128 type corresponding to __float128 (as in GCC headers).
typedef _Complex float __attribute__((mode(TC))) __complex128;

void check() {
  // CHECK: alloca { fp128, fp128 }
  __complex128 tmp;
}
