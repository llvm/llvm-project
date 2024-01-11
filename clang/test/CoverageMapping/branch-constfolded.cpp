// Test that branch regions are not generated for constant-folded conditions.

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-constfolded.cpp %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-constfolded.cpp %s | FileCheck %s -check-prefix=MCDC

// CHECK-LABEL: _Z6fand_0b:
bool fand_0(bool a) {      // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:20 = M:0, C:2
  return false && a;       // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:15 = 0, 0
}                          // CHECK: Branch,File 0, [[@LINE-1]]:19 -> [[@LINE-1]]:20 = #2, (#1 - #2)

// CHECK-LABEL: _Z6fand_1b:
bool fand_1(bool a) {      // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:19 = M:0, C:2
  return a && true;        // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:11 = #1, (#0 - #1)
}                          // CHECK: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:19 = 0, 0

// CHECK-LABEL: _Z6fand_2bb:
bool fand_2(bool a, bool b) {// MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:25 = M:0, C:3
  return false && a && b;  // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:15 = 0, 0
}                          // CHECK: Branch,File 0, [[@LINE-1]]:19 -> [[@LINE-1]]:20 = #4, (#3 - #4)
                           // CHECK: Branch,File 0, [[@LINE-2]]:24 -> [[@LINE-2]]:25 = #2, (#1 - #2)

// CHECK-LABEL: _Z6fand_3bb:
bool fand_3(bool a, bool b) {// MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:24 = M:0, C:3
  return a && true && b;   // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:11 = #3, (#0 - #3)
}                          // CHECK: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:19 = 0, 0
                           // CHECK: Branch,File 0, [[@LINE-2]]:23 -> [[@LINE-2]]:24 = #2, (#1 - #2)

// CHECK-LABEL: _Z6fand_4bb:
bool fand_4(bool a, bool b) {// MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:25 = M:0, C:3
  return a && b && false;  // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:11 = #3, (#0 - #3)
}                          // CHECK: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:16 = #4, (#3 - #4)
                           // CHECK: Branch,File 0, [[@LINE-2]]:20 -> [[@LINE-2]]:25 = 0, 0

// CHECK-LABEL: _Z6fand_5b:
bool fand_5(bool a) {      // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:23 = M:0, C:2
  return false && true;    // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:15 = 0, 0
}                          // CHECK: Branch,File 0, [[@LINE-1]]:19 -> [[@LINE-1]]:23 = 0, 0

// CHECK-LABEL: _Z6fand_6b:
bool fand_6(bool a) {      // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:19 = M:0, C:2
  return true && a;        // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:14 = 0, 0
}                          // CHECK: Branch,File 0, [[@LINE-1]]:18 -> [[@LINE-1]]:19 = #2, (#1 - #2)

// CHECK-LABEL: _Z6fand_7b:
bool fand_7(bool a) {      // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:20 = M:0, C:2
  return a && false;       // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:11 = #1, (#0 - #1)
}                          // CHECK: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:20 = 0, 0

// CHECK-LABEL: _Z5for_0b:
bool for_0(bool a) {       // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:19 = M:0, C:2
  return true || a;        // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:14 = 0, 0
}                          // CHECK: Branch,File 0, [[@LINE-1]]:18 -> [[@LINE-1]]:19 = (#1 - #2), #2

// CHECK-LABEL: _Z5for_1b:
bool for_1(bool a) {       // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:20 = M:0, C:2
  return a || false;       // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:11 = (#0 - #1), #1
}                          // CHECK: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:20 = 0, 0

// CHECK-LABEL: _Z5for_2bb:
bool for_2(bool a, bool b) {// MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:24 = M:0, C:3
  return true || a || b;   // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:14 = 0, 0
}                          // CHECK: Branch,File 0, [[@LINE-1]]:18 -> [[@LINE-1]]:19 = (#3 - #4), #4
                           // CHECK: Branch,File 0, [[@LINE-2]]:23 -> [[@LINE-2]]:24 = (#1 - #2), #2

// CHECK-LABEL: _Z5for_3bb:
bool for_3(bool a, bool b) {// MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:25 = M:0, C:3
  return a || false || b;  // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:11 = (#0 - #3), #3
}                          // CHECK: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:20 = 0, 0
                           // CHECK: Branch,File 0, [[@LINE-2]]:24 -> [[@LINE-2]]:25 = (#1 - #2), #2

// CHECK-LABEL: _Z5for_4bb:
bool for_4(bool a, bool b) {// MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:24 = M:0, C:3
  return a || b || true;   // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:11 = (#0 - #3), #3
}                          // CHECK: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:16 = (#3 - #4), #4
                           // CHECK: Branch,File 0, [[@LINE-2]]:20 -> [[@LINE-2]]:24 = 0, 0

// CHECK-LABEL: _Z5for_5b:
bool for_5(bool a) {       // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:23 = M:0, C:2
  return true || false;    // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:14 = 0, 0
}                          // CHECK: Branch,File 0, [[@LINE-1]]:18 -> [[@LINE-1]]:23 = 0, 0

// CHECK-LABEL: _Z5for_6b:
bool for_6(bool a) {       // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:20 = M:0, C:2
  return false || a;       // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:15 = 0, 0
}                          // CHECK: Branch,File 0, [[@LINE-1]]:19 -> [[@LINE-1]]:20 = (#1 - #2), #2

// CHECK-LABEL: _Z5for_7b:
bool for_7(bool a) {       // MCDC: Decision,File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:19 = M:0, C:2
  return a || true;        // CHECK: Branch,File 0, [[@LINE]]:10 -> [[@LINE]]:11 = (#0 - #1), #1
}                          // CHECK: Branch,File 0, [[@LINE-1]]:15 -> [[@LINE-1]]:19 = 0, 0

// CHECK-LABEL: _Z5for_8b:
bool for_8(bool a) {       // MCDC: Decision,File 0, [[@LINE+3]]:17 -> [[@LINE+3]]:30 = M:0, C:2
                           // CHECK: Branch,File 0, [[@LINE+2]]:17 -> [[@LINE+2]]:21 = 0, 0
                           // CHECK: Branch,File 0, [[@LINE+1]]:25 -> [[@LINE+1]]:30 = 0, 0
  if constexpr (true && false)
      return true;
  else
      return false;
}
