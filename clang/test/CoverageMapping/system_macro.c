// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name system_macro.c -o - %s | FileCheck %s

#ifdef IS_SYSHEADER

#pragma clang system_header
#define Func(x) if (x) {}
#define SomeType int

#else

#define IS_SYSHEADER
#include __FILE__

// CHECK-LABEL: doSomething:
void doSomething(int x) { // CHECK: File 0, [[@LINE]]:25 -> {{[0-9:]+}} = #0
  Func(x); // CHECK: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:7
  return;
  // CHECK: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:11
  SomeType *f; // CHECK: File 0, [[@LINE]]:11 -> {{[0-9:]+}} = 0
}

int main() {}

#endif
