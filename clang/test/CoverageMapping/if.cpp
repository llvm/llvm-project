// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -std=c++1z -triple %itanium_abi_triple -main-file-name if.cpp %s | FileCheck %s

int nop() { return 0; }

// CHECK-LABEL: _Z3foov:
                                // CHECK-NEXT: [[@LINE+1]]:12 -> [[@LINE+6]]:2 = #0
void foo() {                    // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:20 -> [[@LINE+1]]:22 = #2
  if (int j = true ? nop()      // CHECK-NEXT: [[@LINE]]:22 -> [[@LINE]]:27 = #2
                   : nop();     // CHECK-NEXT: [[@LINE]]:22 -> [[@LINE]]:27 = (#0 - #2)
      j)                        // CHECK-NEXT: [[@LINE]]:7 -> [[@LINE]]:8 = #0
    ++j;                        // CHECK-NEXT: [[@LINE-1]]:9 -> [[@LINE]]:5 = #1
}                               // CHECK-NEXT: [[@LINE-1]]:5 -> [[@LINE-1]]:8 = #1

// CHECK-LABEL: main:
int main() {                    // CHECK: File 0, [[@LINE]]:12 -> {{[0-9]+}}:2 = #0
  int i = 0;
                                // CHECK-NEXT: File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #0
                                // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:13 -> [[@LINE+1]]:14 = #1
  if(i == 0) i = 1;             // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE]]:19 = #1

                                // CHECK-NEXT: File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #0
  if(i == 1)                    // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE+1]]:5 = #2
    i = 2;                      // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:10 = #2

                                // CHECK-NEXT: File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #0
  if(i == 0) { i = 1;           // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE]]:14 = #3
    i = 2;                      // CHECK-NEXT: File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:4 = #3
  }
                                // CHECK-NEXT: File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #0
  if(i != 0) {                  // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE]]:14 = #4
    i = 1;                      // CHECK-NEXT: File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:4 = #4
  } else {                      // CHECK-NEXT: Gap,File 0, [[@LINE]]:4 -> [[@LINE]]:10 = (#0 - #4)
    i = 3;                      // CHECK-NEXT: File 0, [[@LINE-1]]:10 -> [[@LINE+1]]:4 = (#0 - #4)
  }

  i = i == 0?                   // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE+1]]:9 = #5
        i + 1 :                 // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = #5
        i + 2;                  // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = (#0 - #5)

                                // CHECK-NEXT: Gap,File 0, [[@LINE+2]]:13 -> [[@LINE+2]]:14 = #6
                                // CHECK-NEXT: File 0, [[@LINE+1]]:14 -> [[@LINE+1]]:20 = #6
  i = i == 0?i + 12:i + 10;     // CHECK-NEXT: File 0, [[@LINE]]:21 -> [[@LINE]]:27 = (#0 - #6)

  return 0;
}
