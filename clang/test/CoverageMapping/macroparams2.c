// RUN: %strip_comments > %t.stripped.c
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroparams2.c %t.stripped.c | FileCheck %s
#define MACRO(REFS, CALLS)  (4 * (CALLS) < (REFS))

struct S {
  int i, j;
};

// CHECK: File 0, [[@LINE+1]]:12 -> [[@LINE+10]]:2 = #0
int main() {
  struct S arr[32] = { 0 };
  int n = 0;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:7 -> [[@LINE+2]]:12 = #0
  // CHECK-NEXT: File 0, [[@LINE+1]]:34 -> [[@LINE+3]]:4 = #1
  if (MACRO(arr[n].j, arr[n].i)) {
    n = 1;
  }
  return n;
}

// CHECK: File 1, 3:29 -> 3:51 = #0
