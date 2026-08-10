// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fcoverage-mcdc -mllvm -system-headers-coverage -emit-llvm-only -o - %s | FileCheck %s --check-prefixes=CHECK,W_SYS
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fcoverage-mcdc -emit-llvm-only -o - %s | FileCheck %s --check-prefixes=CHECK,X_SYS

#ifdef IS_SYSHEADER

#pragma clang system_header
#define CONST 42
#define EXPR1(x) (x)
#define EXPR2(x) ((x) && (x))

#else

#define IS_SYSHEADER
#include __FILE__

// CHECK: _Z5func0i:
int func0(int a) {
  // CHECK: Decision,File 0, [[@LINE+3]]:11 -> [[@LINE+3]]:21 = M:3, C:2
  // W_SYS: Expansion,File 0, [[@LINE+2]]:11 -> [[@LINE+2]]:16 = #0 (Expanded file = 1)
  // X_SYS: Branch,File 0, [[@LINE+1]]:11 -> [[@LINE+1]]:11 = #1, 0 [1,2,0]
  return (CONST && a);
  // CHECK: Branch,File 0, [[@LINE-1]]:20 -> [[@LINE-1]]:21 = #2, (#1 - #2) [2,0,0]
  // W_SYS: Branch,File 1, [[@LINE-16]]:15 -> [[@LINE-16]]:17 = #1, 0 [1,2,0]
}

// CHECK: _Z5func1ii:
int func1(int a, int b) {
  // CHECK: Decision,File 0, [[@LINE+2]]:11 -> [[@LINE+2]]:21 = M:3, C:2
  // CHECK: Branch,File 0, [[@LINE+1]]:11 -> [[@LINE+1]]:12 = (#0 - #1), #1 [1,0,2]
  return (a || EXPR1(b));
  // W_SYS: Expansion,File 0, [[@LINE-1]]:16 -> [[@LINE-1]]:21 = #1 (Expanded file = 1)
  // W_SYS: Branch,File 1, [[@LINE-24]]:18 -> [[@LINE-24]]:21 = (#1 - #2), #2 [2,0,0]
  // X_SYS: Branch,File 0, [[@LINE-3]]:16 -> [[@LINE-3]]:16 = (#1 - #2), #2 [2,0,0]
}

// CHECK: _Z5func2ii:
int func2(int a, int b) {
  // W_SYS: Decision,File 0, [[@LINE+5]]:11 -> [[@LINE+5]]:28 = M:4, C:3
  // X_SYS: Decision,File 0, [[@LINE+4]]:11 -> [[@LINE+4]]:28 = M:3, C:2
  // W_SYS: Expansion,File 0, [[@LINE+3]]:11 -> [[@LINE+3]]:16 = #0 (Expanded file = 1)
  // W_SYS: Expansion,File 0, [[@LINE+2]]:23 -> [[@LINE+2]]:28 = #1 (Expanded file = 2)
  // X_SYS: Branch,File 0, [[@LINE+1]]:11 -> [[@LINE+1]]:11 = #1, (#0 - #1) [1,2,0]
  return (EXPR2(a) && EXPR1(a));
  // W_SYS: Branch,File 1, [[@LINE-35]]:19 -> [[@LINE-35]]:22 = #3, (#0 - #3) [1,3,0]
  // W_SYS: Branch,File 1, [[@LINE-36]]:26 -> [[@LINE-36]]:29 = #4, (#3 - #4) [3,2,0]
  // W_SYS: Branch,File 2, [[@LINE-38]]:18 -> [[@LINE-38]]:21 = #2, (#1 - #2) [2,0,0]
  // X_SYS: Branch,File 0, [[@LINE-4]]:23 -> [[@LINE-4]]:23 = #2, (#1 - #2) [2,0,0]
}

#endif
