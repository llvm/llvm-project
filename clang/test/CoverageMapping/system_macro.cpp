// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -mllvm -system-headers-coverage -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name system_macro.cpp -o - %s | FileCheck %s

#ifdef IS_SYSHEADER

#pragma clang system_header
#define Func(x) if (x) {}
#define SomeType int

#else

#define IS_SYSHEADER
#include __FILE__

// CHECK-LABEL: doSomething
void doSomething(int x) { // CHECK: File 0, [[@LINE]]:25 -> {{[0-9:]+}} = #0
  Func(x); // CHECK: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:7
  return;
  // CHECK: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:11
  SomeType *f; // CHECK: File 0, [[@LINE]]:11 -> {{[0-9:]+}} = 0
}

// CHECK-LABEL: main
int main() { // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+2]]:2 = #0
  Func([] { return true; }());
}

#endif
