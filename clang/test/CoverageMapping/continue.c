// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name continue.c %s | FileCheck %s

int main(void) {                // CHECK: File 0, [[@LINE]]:16 -> [[@LINE+23]]:2 = #0
  int j = 0;                    // CHECK-NEXT: File 0, [[@LINE+3]]:18 -> [[@LINE+3]]:24 = (#0 + #1)
                                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:18 -> [[@LINE+2]]:24 = #1, #0
                                // CHECK-NEXT: File 0, [[@LINE+1]]:26 -> [[@LINE+1]]:29 = #1
  for(int i = 0; i < 20; ++i) { // CHECK: File 0, [[@LINE]]:31 -> [[@LINE+18]]:4 = #1
    if(i < 10) {                // CHECK: File 0, [[@LINE]]:16 -> [[@LINE+14]]:6 = #2
      if(i < 5) {               // CHECK: File 0, [[@LINE]]:17 -> [[@LINE+3]]:8 = #3
        continue;               // CHECK-NEXT: Gap,File 0, [[@LINE]]:18 -> [[@LINE+1]]:9 = 0
        j = 1;                  // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE+1]]:8 = 0
      } else {                  // CHECK: File 0, [[@LINE]]:14 -> [[@LINE+2]]:8 = (#2 - #3)
        j = 2;
      }                         // CHECK-NEXT: Gap,File 0, [[@LINE]]:8 -> [[@LINE+1]]:7 = (#2 - #3)
      j = 3;                    // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+7]]:6 = (#2 - #3)
      if(i < 7) {               // CHECK: File 0, [[@LINE]]:17 -> [[@LINE+3]]:8 = #4
        continue;               // CHECK-NEXT: Gap,File 0, [[@LINE]]:18 -> [[@LINE+1]]:9 = 0
        j = 4;                  // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE+1]]:8 = 0
      } else j = 5;             // CHECK: File 0, [[@LINE]]:14 -> [[@LINE]]:19 = ((#2 - #3) - #4)
                                // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:20 -> [[@LINE+1]]:7 = ((#2 - #3) - #4)
      j = 6;                    // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+1]]:6 = ((#2 - #3) - #4)
    } else                      // CHECK: File 0, [[@LINE+1]]:7 -> [[@LINE+1]]:12 = (#1 - #2)
      j = 7;                    // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE+1]]:5 = ((#1 - #3) - #4)
    j = 8;                      // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+1]]:4 = ((#1 - #3) - #4)
  }
}
