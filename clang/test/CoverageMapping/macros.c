// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macros.c %s | FileCheck %s

#define MACRO return; bar()
#define MACRO_2 bar()
#define MACRO_1 return; MACRO_2
#define MACRO_3 MACRO_2
#define GOTO goto

void bar(void) {}

// CHECK: func
void func(void) {  // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+5]]:2 = #0
  int i = 0;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:8 = #0
  MACRO;       // CHECK-NEXT: File 0, [[@LINE]]:8 -> [[@LINE+2]]:2 = 0
  i = 2;
}
// CHECK-NEXT: File 1, 3:15 -> 3:28 = #0
// CHECK-NEXT: File 1, 3:23 -> 3:28 = 0

// CHECK-NEXT: func2
void func2(void) { // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+5]]:2 = #0
  int i = 0;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:10 = #0
  MACRO_1;     // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE+2]]:2 = 0
  i = 2;
}
// CHECK-NEXT: File 1, 5:17 -> 5:32 = #0
// CHECK-NEXT: Expansion,File 1, 5:25 -> 5:32 = 0
// CHECK-NEXT: File 2, 4:17 -> 4:22 = 0

// CHECK-NEXT: func3
void func3(void) { // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+3]]:2 = #0
  MACRO_2; // CHECK-NEXT: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:10 = #0
  MACRO_2; // CHECK-NEXT: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:10 = #0
}
// CHECK-NEXT: File 1, 4:17 -> 4:22 = #0
// CHECK-NEXT: File 2, 4:17 -> 4:22 = #0

// CHECK-NEXT: func4
void func4(void) { // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+8]]:2 = #0
  int i = 0;
  while (i++ < 10) // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE]]:18 = (#0 + #1)
    if (i < 5) // CHECK: File 0, [[@LINE]]:5 -> [[@LINE+4]]:14 = #1
               // CHECK-NEXT: File 0, [[@LINE-1]]:9 -> [[@LINE-1]]:14 = #1
               // CHECK-NEXT: Branch,File 0, [[@LINE-2]]:9 -> [[@LINE-2]]:14 = #2, (#1 - #2)
               // CHECK-NEXT: Gap,File 0, [[@LINE-3]]:15 -> [[@LINE+1]]:7 = #2
      MACRO_2; // CHECK-NEXT: Expansion,File 0, [[@LINE]]:7 -> [[@LINE]]:14 = #2
}
// CHECK-NEXT: File 1, 4:17 -> 4:22 = #2
// CHECK-NOT: File 1

// CHECK-NEXT: func5
void func5(void) { // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+6]]:2 = #0
  int i = 0;
  if (i > 5) // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = #0
             // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:12 = #1, (#0 - #1)
             // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:13 -> [[@LINE+1]]:5 = #1
    MACRO_3; // CHECK-NEXT: Expansion,File 0, [[@LINE]]:5 -> [[@LINE]]:12 = #1
}
// CHECK-NEXT: Expansion,File 1, 6:17 -> 6:24 = #1
// CHECK-NEXT: File 2, 4:17 -> 4:22 = #1

// CHECK-NEXT: func6
void func6(unsigned count) { // CHECK-NEXT: File 0, [[@LINE]]:28 -> [[@LINE+6]]:2 = #0
begin:                       // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE+5]]:2 = #1
    if (count--)             // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:16 = #1
                             // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:9 -> [[@LINE-1]]:16 = #2, (#1 - #2)
                             // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:17 -> [[@LINE+1]]:9 = #2
        GOTO begin;          // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:19 = #2
}
// CHECK-NEXT: Expansion,File 0, [[@LINE-2]]:9 -> [[@LINE-2]]:13 = #2
// CHECK-NEXT: File 1, 7:14 -> 7:18 = #2

// Regression test for gap region between macros.
// CHECK-NEXT: func7
int k, l;
#define m(e) e##e
void func7(void) { // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+6]]:2 = #0
  int kk,ll;       // CHECK-NEXT: File 0, [[@LINE+1]]:7 -> [[@LINE+1]]:8 = #0
  if (k)           // CHECK-NEXT: Branch,File 0, [[@LINE]]:7 -> [[@LINE]]:8 = #1
    m(k);          // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:9 -> [[@LINE]]:5 = #1
  else             // CHECK-NEXT: Expansion,File 0, [[@LINE-1]]:5 -> [[@LINE-1]]:6 = #1
    l = m(l);      // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:7 -> [[@LINE]]:5 = (#0 - #1)
}                  // CHECK-NEXT: File 0, [[@LINE-1]]:5 -> [[@LINE-1]]:10 = (#0 - #1)
                   // CHECK-NEXT: Expansion,File 0, [[@LINE-2]]:9 -> [[@LINE-2]]:10 = (#0 - #1)
                   // CHECK-NEXT: File 1, [[@LINE-9]]:14 -> [[@LINE-9]]:17 = #1
                   // CHECK-NEXT: File 1, [[@LINE-10]]:14 -> [[@LINE-10]]:18 = #0
                   // CHECK-NEXT: File 2, [[@LINE-11]]:14 -> [[@LINE-11]]:17 = (#0 - #1)
                   // CHECK-NEXT: File 2, [[@LINE-12]]:14 -> [[@LINE-12]]:15 = (#0 - #1)

int main(int argc, const char *argv[]) {
  func();
  func2();
  func3();
  func4();
}
