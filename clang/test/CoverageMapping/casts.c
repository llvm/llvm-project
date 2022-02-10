// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name casts.c %s | FileCheck %s

int main(void) {                                               // CHECK: File 0, [[@LINE]]:16 -> [[@LINE+4]]:2 = #0
                                                               // CHECK: File 0, [[@LINE+1]]:41 -> [[@LINE+1]]:54 = #1
  int window_size = (sizeof(int) <= 2 ? (unsigned)512 : 1024); // CHECK-NEXT: File 0, [[@LINE]]:57 -> [[@LINE]]:61 = (#0 - #1)
  return 0;
}




