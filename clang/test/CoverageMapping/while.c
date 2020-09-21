// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name loops.cpp %s | FileCheck %s

// CHECK: main
int main() {                        // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+8]]:2 = #0
  int j = 0;                        // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:14 = (#0 + #1)
  while(j < 5) ++j;                 // CHECK-NEXT: File 0, [[@LINE]]:15 -> [[@LINE]]:16 = #1
  j = 0;                            // CHECK-NEXT: File 0, [[@LINE-1]]:16 -> [[@LINE-1]]:19 = #1
  while                             // CHECK-NEXT: File 0, [[@LINE+1]]:5 -> [[@LINE+1]]:10 = (#0 + #2)
   (j < 5)                          // CHECK-NEXT: Gap,File 0, [[@LINE]]:11 -> [[@LINE+1]]:6 = #2
     ++j;                           // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE]]:9 = #2
  return 0;
}
