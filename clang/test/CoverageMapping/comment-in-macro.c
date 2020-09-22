// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

#define x1 "" // ...
#define x2 return 0
// CHECK: main
int main() { // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+3]]:2 = #0
  x1;        // CHECK-NEXT: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:5 = #0
  x2;        // CHECK-NEXT: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:5 = #0
}
// CHECK-NEXT: File 1, 3:12 -> 3:14 = #0
// CHECK-NEXT: File 2, 4:12 -> 4:20 = #0
