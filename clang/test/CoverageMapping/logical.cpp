// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name logical.cpp %s | FileCheck %s

int main() {                        // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+15]]:2 = #0
  bool bt = true;
  bool bf = false;
  bool a = bt && bf;                // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE]]:14 = #0
                                    // CHECK-NEXT: File 0, [[@LINE-1]]:18 -> [[@LINE-1]]:20 = #1

  a = bt &&                         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #0
      bf;                           // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #2

  a = bf || bt;                     // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #0
                                    // CHECK-NEXT: File 0, [[@LINE-1]]:13 -> [[@LINE-1]]:15 = #3

  a = bf ||                         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #0
      bt;                           // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #4
  return 0;
}
