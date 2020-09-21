// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroparams.c %s | FileCheck %s

// CHECK: main
// CHECK-NEXT: File 0, {{[0-9]+}}:12 -> {{[0-9]+}}:2 = #0
// CHECK-NEXT: Expansion,File 0, {{[0-9]+}}:3 -> {{[0-9]+}}:8 = #0

// CHECK-NEXT: File 1, [[@LINE+2]]:18 -> [[@LINE+2]]:27 = #0
// CHECK-NEXT: Expansion,File 1, [[@LINE+1]]:18 -> [[@LINE+1]]:24 = #0
#define MACRO(X) MACRO2(x)
// CHECK-NEXT: File 2, [[@LINE+1]]:20 -> [[@LINE+1]]:28 = #0
#define MACRO2(X2) (X2 + 2)

int main() {
  int x = 0;
  MACRO(x);
  return 0;
}
