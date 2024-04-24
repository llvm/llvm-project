// Test that branch regions are generated for return paths when calls-end-coverage-region is set.

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-return.cpp %s | FileCheck %s -check-prefix=WITHOUT
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -mllvm -calls-end-coverage-region -dump-coverage-mapping -emit-llvm-only -main-file-name branch-return.cpp %s | FileCheck %s -check-prefix=WITH

// CHECK-LABEL: _Z5func1ii:
// WITHOUT: File 0, [[@LINE+2]]:26 -> [[@LINE+4]]:2 = #0
// WITH: File 0, [[@LINE+1]]:26 -> [[@LINE+3]]:2 = #0
bool func1(int a, int b) {
  return (a | b) < 100;
}

// CHECK-LABEL: _Z5func2ii:
// WITHOUT: File 0, [[@LINE+2]]:26 -> [[@LINE+4]]:2 = #0
// WITH: File 0, [[@LINE+1]]:26 -> [[@LINE+3]]:2 = #0
bool func2(int a, int b) {
  return (a ^ b) > 10;
}

// CHECK-LABEL: _Z5func3ii:
// WITHOUT: File 0, [[@LINE+6]]:26 -> [[@LINE+10]]:2 = #0
// WITH: File 0, [[@LINE+5]]:26 -> [[@LINE+9]]:2 = #0
// WITH: Gap,File 0, [[@LINE+5]]:27 -> [[@LINE+6]]:3 = 0
// WITH: File 0, [[@LINE+5]]:3 -> [[@LINE+5]]:26 = 0
// WITH: Gap,File 0, [[@LINE+4]]:27 -> [[@LINE+5]]:3 = 0
// WITH: File 0, [[@LINE+4]]:3 -> [[@LINE+4]]:22 = 0
bool func3(int a, int b) {
  bool val1 = func1(a, b);
  bool val2 = func2(a, b);
  return val1 || val2;
}
