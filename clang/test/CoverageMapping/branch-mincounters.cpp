// Test to ensure right number of counters are allocated and used for nested
// logical operators on branch conditions for branch coverage.

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-logical-mixed.cpp %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-logical-mixed.cpp %s | FileCheck %s


// CHECK-LABEL: _Z5func1ii:
bool func1(int a, int b) {
  bool b0 = a <= b;
  bool b1 = a == b;
  bool b2 = a >= b;

  // This should allocate a RHS branch counter on b2 (counter #3).
  bool c = b0 && (b1 || b2);
  // CHECK: Branch,File 0, [[@LINE-1]]:12 -> [[@LINE-1]]:14 = #1, (#0 - #1)
  // CHECK: Branch,File 0, [[@LINE-2]]:19 -> [[@LINE-2]]:21 = (#1 - #2), #2
  // CHECK: Branch,File 0, [[@LINE-3]]:25 -> [[@LINE-3]]:27 = (#2 - #3), #3

  return c;
}

// CHECK-LABEL: _Z5func2ii:
bool func2(int a, int b) {
  bool b0 = a <= b;
  bool b1 = a == b;
  bool b2 = a >= b;

  // This should allocate a RHS branch counter on b1 and b2 (counters #2, #4)
  // This could possibly be further optimized through counter reuse (future).
  bool c = (b0 && b1) || b2;
  // CHECK: Branch,File 0, [[@LINE-1]]:13 -> [[@LINE-1]]:15 = #3, (#0 - #3)
  // CHECK: Branch,File 0, [[@LINE-2]]:19 -> [[@LINE-2]]:21 = #4, (#3 - #4)
  // CHECK: Branch,File 0, [[@LINE-3]]:26 -> [[@LINE-3]]:28 = (#1 - #2), #2

  return c;
}

// CHECK-LABEL: _Z5func3ii:
bool func3(int a, int b) {
  bool b0 = a <= b;
  bool b1 = a == b;
  bool b2 = a >= b;
  bool b3 = a < b;

  // This should allocate a RHS branch counter on b1 and b3 (counters #3, #5)
  // This could possibly be further optimized through counter reuse (future).
  bool c = (b0 || b1) && (b2 || b3);
  // CHECK: Branch,File 0, [[@LINE-1]]:13 -> [[@LINE-1]]:15 = (#0 - #2), #2
  // CHECK: Branch,File 0, [[@LINE-2]]:19 -> [[@LINE-2]]:21 = (#2 - #3), #3
  // CHECK: Branch,File 0, [[@LINE-3]]:27 -> [[@LINE-3]]:29 = (#1 - #4), #4
  // CHECK: Branch,File 0, [[@LINE-4]]:33 -> [[@LINE-4]]:35 = (#4 - #5), #5

  return c;
}
