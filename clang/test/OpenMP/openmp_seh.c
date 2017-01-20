// RUN: %clang_cc1 -verify -triple x86_64-pc-windows-msvc19.0.0 -fopenmp -fms-compatibility -x c++ -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
extern "C" {
void __cpuid(int[4], int);
}

// CHECK-LABEL: @main
int main(void) {
  __try {
    int info[4];
    __cpuid(info, 1);
  } __except (1) {
  }

  return 0;
}

