// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name return.c %s | FileCheck %s

// CHECK: func
void func() {                   // CHECK: File 0, [[@LINE]]:13 -> [[@LINE+3]]:2 = #0
  return;                       // CHECK-NEXT: Gap,File 0, [[@LINE]]:10 -> [[@LINE+1]]:3 = 0
  int i = 0;                    // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:2 = 0
}

                                // CHECK-NEXT: func2
void func2() {                  // CHECK-NEXT: File 0, [[@LINE]]:14 -> {{[0-9]+}}:2 = #0
                                // CHECK-NEXT: File 0, [[@LINE+3]]:18 -> [[@LINE+3]]:24 = ((#0 + #1) - #2)
                                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:18 -> [[@LINE+2]]:24 = #1, (#0 - #2)
                                // CHECK-NEXT: File 0, [[@LINE+1]]:26 -> [[@LINE+1]]:29 = (#1 - #2)
  for(int i = 0; i < 10; ++i) { // CHECK: File 0, [[@LINE]]:31 -> {{[0-9]+}}:4 = #1
                                // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+1]]:13 = #1
    if(i > 2) {                 // CHECK: File 0, [[@LINE]]:15 -> [[@LINE+2]]:6 = #2
      return;                   // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:6 -> [[@LINE+3]]:5 = (#1 - #2)
    }                           // CHECK-NEXT: File 0, [[@LINE+2]]:5 -> {{[0-9]+}}:4 = (#1 - #2)
                                // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+1]]:14 = (#1 - #2)
    if(i == 3) {                // CHECK: File 0, [[@LINE]]:16 -> [[@LINE+2]]:6 = #3
      int j = 1;
    } else {                    // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = ((#1 - #2) - #3)
      int j = 2;
    }
  }
}

                               // CHECK-NEXT: func3
void func3(int x) {            // CHECK-NEXT: File 0, [[@LINE]]:19 -> {{[0-9]+}}:2 = #0
                               // CHECK-NEXT: File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:11 = #0
  if(x > 5) {                  // CHECK: File 0, [[@LINE]]:13 -> [[@LINE+7]]:4 = #1
    while(x >= 9) {            // CHECK-NEXT: File 0, [[@LINE]]:11 -> [[@LINE]]:17 = #1
      return;                  // CHECK: File 0, [[@LINE-1]]:19 -> [[@LINE+3]]:6 = #2
                               // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:7 = 0
      --x;                     // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+1]]:6 = 0
    }                          // CHECK-NEXT: Gap,File 0, [[@LINE]]:6 -> [[@LINE+1]]:5 = (#1 - #2)
    int i = 0;                 // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+1]]:4 = (#1 - #2)
  }                            // CHECK-NEXT: Gap,File 0, [[@LINE]]:4 -> [[@LINE+1]]:3 = (#0 - #2)
  int j = 0;                   // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:2 = (#0 - #2)
}

int main(int argc, const char *argv[]) {
  func();
  func2();
  func3(10);
}
