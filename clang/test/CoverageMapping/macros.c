// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macros.c %s | FileCheck %s

#define MACRO return; bar()
#define MACRO_2 bar()
#define MACRO_1 return; MACRO_2
#define MACRO_3 MACRO_2
#define GOTO goto

void bar() {}

// CHECK: func
void func() {  // CHECK-NEXT: File 0, [[@LINE]]:13 -> [[@LINE+5]]:2 = #0
  int i = 0;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:8 = #0
  MACRO;       // CHECK-NEXT: File 0, [[@LINE]]:8 -> [[@LINE+2]]:2 = 0
  i = 2;
}
// CHECK-NEXT: File 1, 3:15 -> 3:28 = #0
// CHECK-NEXT: File 1, 3:23 -> 3:28 = 0

// CHECK-NEXT: func2
void func2() { // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+5]]:2 = #0
  int i = 0;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:10 = #0
  MACRO_1;     // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE+2]]:2 = 0
  i = 2;
}
// CHECK-NEXT: File 1, 5:17 -> 5:32 = #0
// CHECK-NEXT: Expansion,File 1, 5:25 -> 5:32 = 0
// CHECK-NEXT: File 2, 4:17 -> 4:22 = 0

// CHECK-NEXT: func3
void func3() { // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+3]]:2 = #0
  MACRO_2; // CHECK-NEXT: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:10 = #0
  MACRO_2; // CHECK-NEXT: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:10 = #0
}
// CHECK-NEXT: File 1, 4:17 -> 4:22 = #0
// CHECK-NEXT: File 2, 4:17 -> 4:22 = #0

// CHECK-NEXT: func4
void func4() { // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+6]]:2 = #0
  int i = 0;
  while (i++ < 10) // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE]]:18 = (#0 + #1)
    if (i < 5) // CHECK: File 0, [[@LINE]]:5 -> [[@LINE+2]]:14 = #1
               // CHECK-NEXT: File 0, [[@LINE-1]]:9 -> [[@LINE-1]]:14 = #1
      MACRO_2; // CHECK-NEXT: Expansion,File 0, [[@LINE]]:7 -> [[@LINE]]:14 = #2
}
// CHECK-NEXT: File 1, 4:17 -> 4:22 = #2
// CHECK-NOT: File 1

// CHECK-NEXT: func5
void func5() { // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+4]]:2 = #0
  int i = 0;
  if (i > 5) // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = #0
    MACRO_3; // CHECK-NEXT: Expansion,File 0, [[@LINE]]:5 -> [[@LINE]]:12 = #1
}
// CHECK-NEXT: Expansion,File 1, 6:17 -> 6:24 = #1
// CHECK-NEXT: File 2, 4:17 -> 4:22 = #1

// CHECK-NEXT: func6
void func6(unsigned count) { // CHECK-NEXT: File 0, [[@LINE]]:28 -> [[@LINE+4]]:2 = #0
begin:                       // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE+3]]:2 = #1
    if (count--)             // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:16 = #1
        GOTO begin;          // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:19 = #2
}
// CHECK-NEXT: Expansion,File 0, [[@LINE-2]]:9 -> [[@LINE-2]]:13 = #2
// CHECK-NEXT: File 1, 7:14 -> 7:18 = #2

int main(int argc, const char *argv[]) {
  func();
  func2();
  func3();
  func4();
}
