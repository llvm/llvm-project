// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name logical.cpp %s | FileCheck %s

int main() {                        // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+10]]:2 = #0
  bool bt = true;
  bool bf = false;
  bool a = bt && bf;                // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE]]:20 = #1
  a = bt &&
      bf;                           // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #2
  a = bf || bt;                     // CHECK-NEXT: File 0, [[@LINE]]:13 -> [[@LINE]]:15 = #3
  a = bf ||
      bt;                           // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #4
  return 0;
}
