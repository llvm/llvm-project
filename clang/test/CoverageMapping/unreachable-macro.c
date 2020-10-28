// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

#define WHILE while (0) {}

// CHECK: counters_in_macro_following_unreachable
void counters_in_macro_following_unreachable() {
  // CHECK-NEXT: File 0, [[@LINE-1]]:48 -> {{[0-9]+}}:2 = #0
  return;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:3 -> [[@LINE+2]]:8 = 0
  // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+2]]:2 = 0
  WHILE
}
// CHECK-NEXT: File 1, 3:15 -> 3:27 = 0
// CHECK-NEXT: File 1, 3:22 -> 3:23 = #1
// CHECK-NEXT: File 1, 3:25 -> 3:27 = #1
