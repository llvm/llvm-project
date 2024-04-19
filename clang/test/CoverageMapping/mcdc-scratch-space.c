// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c99 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

// CHECK: builtin_macro0:
int builtin_macro0(int a) {
  // CHECK: Decision,File 0, [[@LINE+1]]:11 -> [[@LINE+2]]:15 = M:0, C:2
  return (__LINE__ // CHECK: Branch,File 0, [[@LINE]]:11 -> [[@LINE]]:11 = 0, 0 [1,2,0]
          && a); //   CHECK: Branch,File 0, [[@LINE]]:14 -> [[@LINE]]:15 = #2, (#1 - #2) [2,0,0]
}

// CHECK: builtin_macro1:
int builtin_macro1(int a) {
  // CHECK: Decision,File 0, [[@LINE+1]]:11 -> [[@LINE+2]]:22 = M:0, C:2
  return (a // CHECK: Branch,File 0, [[@LINE]]:11 -> [[@LINE]]:12 = (#0 - #1), #1 [1,0,2]
          || __LINE__); // CHECK: Branch,File 0, [[@LINE]]:14 -> [[@LINE]]:14 = 0, 0 [2,0,0]
}

#define PRE(x) pre_##x

// CHECK: pre0:
int pre0(int pre_a, int b_post) {
  // CHECK: Decision,File 0, [[@LINE+2]]:11 -> [[@LINE+3]]:20 = M:0, C:2
  // CHECK: Expansion,File 0, [[@LINE+1]]:11 -> [[@LINE+1]]:14 = #0 (Expanded file = 1)
  return (PRE(a)
          && b_post);
  // CHECK: Branch,File 0, [[@LINE-1]]:14 -> [[@LINE-1]]:20 = #2, (#1 - #2) [2,0,0]
  // CHECK: Branch,File 1, [[@LINE-9]]:16 -> [[@LINE-9]]:22 = #1, (#0 - #1) [1,2,0]
}

#define POST(x) x##_post

// CHECK: post0:
int post0(int pre_a, int b_post) {
  // CHECK: Decision,File 0, [[@LINE+2]]:11 -> [[@LINE+3]]:18 = M:0, C:2
  // CHECK: Branch,File 0, [[@LINE+1]]:11 -> [[@LINE+1]]:16 = (#0 - #1), #1 [1,0,2]
  return (pre_a
          || POST(b));
  // CHECK: Expansion,File 0, [[@LINE-1]]:14 -> [[@LINE-1]]:18 = #1 (Expanded file = 1)
  // CHECK: Branch,File 1, [[@LINE-9]]:17 -> [[@LINE-9]]:20 = (#1 - #2), #2 [2,0,0]
}
