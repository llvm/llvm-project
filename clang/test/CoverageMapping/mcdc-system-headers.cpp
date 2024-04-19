// RUN: %clang_cc1 -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping  -fcoverage-mcdc -mllvm -system-headers-coverage -emit-llvm-only -o - %s | FileCheck %s

// Will crash w/o -system-headers-coverage
// RUN: not --crash %clang_cc1 -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fcoverage-mcdc -emit-llvm-only -o - %s
// REQUIRES: asserts

#ifdef IS_SYSHEADER

#pragma clang system_header
#define CONST 42
#define EXPR1(x) (x)
#define EXPR2(x) ((x) * (x))

#else

#define IS_SYSHEADER
#include __FILE__

// CHECK: _Z5func0i:
int func0(int a) {
  // CHECK: Decision,File 0, [[@LINE+2]]:11 -> [[@LINE+2]]:21 = M:0, C:2
  // CHECK: Expansion,File 0, [[@LINE+1]]:11 -> [[@LINE+1]]:16 = #0 (Expanded file = 1)
  return (CONST && a);
  // CHECK: Branch,File 0, [[@LINE-1]]:20 -> [[@LINE-1]]:21 = #2, (#1 - #2) [2,0,0]
  // CHECK: Branch,File 1, [[@LINE-15]]:15 -> [[@LINE-15]]:17 = 0, 0 [1,2,0]
}

// CHECK: _Z5func1ii:
int func1(int a, int b) {
  // CHECK: Decision,File 0, [[@LINE+2]]:11 -> [[@LINE+2]]:21 = M:0, C:2
  // CHECK: Branch,File 0, [[@LINE+1]]:11 -> [[@LINE+1]]:12 = (#0 - #1), #1 [1,0,2]
  return (a || EXPR1(b));
  // CHECK: Expansion,File 0, [[@LINE-1]]:16 -> [[@LINE-1]]:21 = #1 (Expanded file = 1)
  // CHECK: Branch,File 1, [[@LINE-23]]:18 -> [[@LINE-23]]:21 = (#1 - #2), #2 [2,0,0]
}

// CHECK: _Z5func2ii:
int func2(int a, int b) {
  // Decision,File 0, [[@LINE+3]]:11 -> [[@LINE+3]]:28 = M:0, C:2
  // Expansion,File 0, [[@LINE+2]]:11 -> [[@LINE+2]]:16 = #0 (Expanded file = 1)
  // Expansion,File 0, [[@LINE+1]]:23 -> [[@LINE+1]]:28 = #1 (Expanded file = 2)
  return (EXPR2(a) && EXPR1(a));
  // CHECK: Branch,File 1, [[@LINE-31]]:18 -> [[@LINE-31]]:29 = #1, (#0 - #1) [1,2,0]
  // CHECK: Branch,File 2, [[@LINE-33]]:18 -> [[@LINE-33]]:21 = #2, (#1 - #2) [2,0,0]
}

#endif
