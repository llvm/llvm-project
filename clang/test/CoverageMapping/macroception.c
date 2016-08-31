// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroception.c %s | FileCheck %s

#define M2 {
#define M1 M2
#define M22 }
#define M11 M22

// CHECK-LABEL: main:
// CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:12 -> [[@LINE+2]]:14 = #0
// CHECK-NEXT: File 0, [[@LINE+1]]:14 -> [[@LINE+3]]:2 = #0
int main() M1
  return 0;
}
// CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0
// CHECK-NEXT: File 2, 3:12 -> 3:13 = #0

// CHECK-LABEL: func2:
// CHECK-NEXT: File 0, [[@LINE+2]]:14 -> [[@LINE+4]]:4 = #0
// CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:1 -> [[@LINE+3]]:4 = #0
void func2() {
  int x = 0;
M11
// CHECK-NEXT: Expansion,File 1, 6:13 -> 6:16 = #0
// CHECK-NEXT: File 2, 5:13 -> 5:14 = #0

// CHECK-LABEL: func3:
// CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:14 -> [[@LINE+3]]:16 = #0
// CHECK-NEXT: File 0, [[@LINE+2]]:16 -> [[@LINE+4]]:4 = #0
// CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:1 -> [[@LINE+3]]:4 = #0
void func3() M1
  int x = 0;
M11
// CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0
// CHECK-NEXT: Expansion,File 2, 6:13 -> 6:16 = #0
// CHECK-NEXT: File 3, 3:12 -> 3:13 = #0
// CHECK-NEXT: File 4, 5:13 -> 5:14 = #0

// CHECK-LABEL: func4:
// CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:14 -> [[@LINE+3]]:16 = #0
// CHECK-NEXT: File 0, [[@LINE+2]]:16 -> [[@LINE+2]]:20 = #0
// CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:17 -> [[@LINE+1]]:20 = #0
void func4() M1 M11
// CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0
// CHECK-NEXT: Expansion,File 2, 6:13 -> 6:16 = #0
// CHECK-NEXT: File 3, 3:12 -> 3:13 = #0
// CHECK-NEXT: File 4, 5:13 -> 5:14 = #0
