// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

#define ifc if

// CHECK: main
// CHECK-NEXT: File 0, {{[0-9]+}}:40 -> [[END:[0-9]+]]:2 = #0
int main(int argc, const char *argv[]) {
  // CHECK: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:6 = #0
  ifc(1) return 0;
  // Expansion,File 0, [[@LINE+2]]:3 -> [[@LINE+2]]:6 = (#0 - #1)
  // File 0, [[@LINE+1]]:6 -> [[END]]:2 = (#0 - #1)
  ifc(1) return 0;
  return 0;
}
