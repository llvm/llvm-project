// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name break.c %s | FileCheck %s

int main() {         // CHECK: File 0, [[@LINE]]:12 -> {{[0-9]+}}:2 = #0
  int cnt = 0;       // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = #0
  while(cnt < 100) { // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+3]]:4 = #1
    break;
    ++cnt;           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+1]]:4 = 0
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = #0
  while(cnt < 100) { // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+6]]:4 = #2
    {
      break;
      ++cnt;         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+3]]:4 = 0
    }
    ++cnt;
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = ((#0 + #3) - #4)
  while(cnt < 100) { // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+7]]:4 = #3
                     // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+1]]:16 = #3
    if(cnt == 0) {   // CHECK: File 0, [[@LINE]]:18 -> [[@LINE+3]]:6 = #4
      break;
      ++cnt;         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+1]]:6 = 0
    }
    ++cnt;           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+1]]:4 = (#3 - #4)
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = (#0 + #6)
  while(cnt < 100) { // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+8]]:4 = #5
                     // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+1]]:16 = #5
    if(cnt == 0) {   // CHECK: File 0, [[@LINE]]:18 -> [[@LINE+2]]:6 = #6
      ++cnt;         // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = (#5 - #6)
    } else {         // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = (#5 - #6)
      break;
    }
    ++cnt;
  }
}
