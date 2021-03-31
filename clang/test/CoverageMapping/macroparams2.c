// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroparams2.c %s | FileCheck %s

#define MACRO(REFS, CALLS)  (4 * (CALLS) < (REFS))

struct S {
  int i, j;
};

// CHECK: File 0, [[@LINE+1]]:12 -> [[@LINE+11]]:2 = #0
int main() {
  struct S arr[32] = { 0 };
  int n = 0;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:7 -> [[@LINE+3]]:12 = #0
  // CHECK-NEXT: Gap,File 0, [[@LINE+2]]:33 -> [[@LINE+2]]:34 = #1
  // CHECK-NEXT: File 0, [[@LINE+1]]:34 -> [[@LINE+3]]:4 = #1
  if (MACRO(arr[n].j, arr[n].i)) {
    n = 1;
  }
  return n;
}

// CHECK: File 1, 3:29 -> 3:51 = #0
