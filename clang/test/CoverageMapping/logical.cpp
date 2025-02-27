// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name logical.cpp %s | FileCheck %s
// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name logical.cpp %s | FileCheck %s -check-prefix=MCDC

int main() {                        // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+23]]:2 = #0
  bool bt = true;
  bool bf = false;                  // MCDC: Decision,File 0, [[@LINE+1]]:12 -> [[@LINE+1]]:20 = M:3, C:2
  bool a = bt && bf;                // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE]]:14 = #0
                                    // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:12 -> [[@LINE-1]]:14 = #1, (#0 - #1)
                                    // CHECK-NEXT: File 0, [[@LINE-2]]:18 -> [[@LINE-2]]:20 = #1
                                    // CHECK-NEXT: Branch,File 0, [[@LINE-3]]:18 -> [[@LINE-3]]:20 = #2, (#1 - #2)
                                    // MCDC: Decision,File 0, [[@LINE+1]]:7 -> [[@LINE+2]]:9 = M:6, C:2
  a = bt &&                         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #0
      bf;                           // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:9 = #3, (#0 - #3)
                                    // CHECK-NEXT: File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:9 = #3
                                    // CHECK-NEXT: Branch,File 0, [[@LINE-2]]:7 -> [[@LINE-2]]:9 = #4, (#3 - #4)
                                    // MCDC: Decision,File 0, [[@LINE+1]]:7 -> [[@LINE+1]]:15 = M:9, C:2
  a = bf || bt;                     // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #0
                                    // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:9 = (#0 - #5), #5
                                    // CHECK-NEXT: File 0, [[@LINE-2]]:13 -> [[@LINE-2]]:15 = #5
                                    // CHECK-NEXT: Branch,File 0, [[@LINE-3]]:13 -> [[@LINE-3]]:15 = (#5 - #6), #6
                                    // MCDC: Decision,File 0, [[@LINE+1]]:7 -> [[@LINE+2]]:9 = M:12, C:2
  a = bf ||                         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #0
      bt;                           // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:9 = (#0 - #7), #7
                                    // CHECK-NEXT: File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:9 = #7
                                    // CHECK-NEXT: Branch,File 0, [[@LINE-2]]:7 -> [[@LINE-2]]:9 = (#7 - #8), #8
  return 0;
}
