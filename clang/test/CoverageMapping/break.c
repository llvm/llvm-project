// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name break.c %s | FileCheck %s

int main(void) {     // CHECK: File 0, [[@LINE]]:16 -> {{[0-9]+}}:2 = #0
  int cnt = 0;       // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = #0
  while(cnt < 100) { // CHECK: File 0, [[@LINE]]:20 -> [[@LINE+3]]:4 = #1
    break;           // CHECK-NEXT: Gap,File 0, [[@LINE]]:11 -> [[@LINE+1]]:5 = 0
    ++cnt;           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+1]]:4 = 0
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = #0
  while(cnt < 100) { // CHECK: File 0, [[@LINE]]:20 -> [[@LINE+6]]:4 = #2
    {
      break;         // CHECK: Gap,File 0, [[@LINE]]:13 -> [[@LINE+1]]:7 = 0
      ++cnt;         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+3]]:4 = 0
    }                // CHECK: Gap,File 0, [[@LINE]]:6 -> [[@LINE+1]]:5 = 0
    ++cnt;
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = ((#0 + #3) - #4)
  while(cnt < 100) { // CHECK: File 0, [[@LINE]]:20 -> [[@LINE+7]]:4 = #3
                     // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+1]]:16 = #3
    if(cnt == 0) {   // CHECK: File 0, [[@LINE]]:18 -> [[@LINE+3]]:6 = #4
      break;         // CHECK: Gap,File 0, [[@LINE]]:13 -> [[@LINE+1]]:7 = 0
      ++cnt;         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+1]]:6 = 0
    }                // CHECK: Gap,File 0, [[@LINE]]:6 -> [[@LINE+1]]:5 = (#3 - #4)
    ++cnt;           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+1]]:4 = (#3 - #4)
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = (#0 + #6)
  while(cnt < 100) { // CHECK: File 0, [[@LINE]]:20 -> [[@LINE+8]]:4 = #5
                     // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+1]]:16 = #5
    if(cnt == 0) {   // CHECK: File 0, [[@LINE]]:18 -> [[@LINE+2]]:6 = #6
      ++cnt;         // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = (#5 - #6)
    } else {         // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = (#5 - #6)
      break;
    }
    ++cnt;
  }
}

// CHECK-LABEL: break_continue_in_increment:
/*
  52:41 -> 57:2 = #0
  53:10 -> 53:11 = ((#0 + #1) + #2)
  Branch,File 0, 53:10 -> 53:11 = #1, 0

  53:13 -> 56:4 = #1


  

*/

  // CHECK: [[@LINE+6]]:20 -> [[@LINE+6]]:21 = #2
  // CHECK: [[@LINE+5]]:23 -> [[@LINE+5]]:28 = #3
  // CHECK: [[@LINE+4]]:35 -> [[@LINE+4]]:43 = (#2 - #3)
  // CHECK: [[@LINE+4]]:7 -> [[@LINE+4]]:8 = #2
void break_continue_in_increment(int x) {
  while (1) {
    for (;; ({ if (x) break; else continue; }))
      ;
  }
}
