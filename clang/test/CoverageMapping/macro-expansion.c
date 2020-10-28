// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macro-expansion.c %s | FileCheck %s

// CHECK: func
// CHECK:      File 1, [[@LINE+5]]:12 -> [[@LINE+5]]:38 = #0
// CHECK-NEXT: File 1, [[@LINE+4]]:15 -> [[@LINE+4]]:28 = (#0 + #2)
// CHECK-NEXT: File 1, [[@LINE+3]]:21 -> [[@LINE+3]]:22 = (#0 + #2)
// CHECK: File 1, [[@LINE+2]]:24 -> [[@LINE+2]]:26 = #3
// CHECK-NEXT: File 1, [[@LINE+1]]:36 -> [[@LINE+1]]:37 = (#0 + #2)
#define M1 do { if (0) {} } while (0)
// CHECK-NEXT: File 2, [[@LINE+10]]:15 -> [[@LINE+10]]:41 = #0
// CHECK-NEXT: File 2, [[@LINE+9]]:18 -> [[@LINE+9]]:31 = (#0 + #4)
// CHECK-NEXT: File 2, [[@LINE+8]]:24 -> [[@LINE+8]]:25 = (#0 + #4)
// CHECK: File 2, [[@LINE+7]]:27 -> [[@LINE+7]]:29 = #5
// CHECK-NEXT: File 2, [[@LINE+6]]:39 -> [[@LINE+6]]:40 = (#0 + #4)
// CHECK-NEXT: File 3, [[@LINE+5]]:15 -> [[@LINE+5]]:41 = #0
// CHECK-NEXT: File 3, [[@LINE+4]]:18 -> [[@LINE+4]]:31 = (#0 + #6)
// CHECK-NEXT: File 3, [[@LINE+3]]:24 -> [[@LINE+3]]:25 = (#0 + #6)
// CHECK: File 3, [[@LINE+2]]:27 -> [[@LINE+2]]:29 = #7
// CHECK-NEXT: File 3, [[@LINE+1]]:39 -> [[@LINE+1]]:40 = (#0 + #6)
#define M2(x) do { if (x) {} } while (0)
// CHECK-NEXT: File 4, [[@LINE+4]]:15 -> [[@LINE+4]]:38 = #0
// CHECK-NEXT: File 4, [[@LINE+3]]:18 -> [[@LINE+3]]:28 = (#0 + #8)
// CHECK-NEXT: Expansion,File 4, [[@LINE+2]]:20 -> [[@LINE+2]]:22 = (#0 + #8)
// CHECK-NEXT: File 4, [[@LINE+1]]:36 -> [[@LINE+1]]:37 = (#0 + #8)
#define M3(x) do { M2(x); } while (0)
// CHECK-NEXT: File 5, [[@LINE+3]]:15 -> [[@LINE+3]]:27 = #0
// CHECK-NEXT: File 5, [[@LINE+2]]:16 -> [[@LINE+2]]:19 = #0
// CHECK-NEXT: File 5, [[@LINE+1]]:23 -> [[@LINE+1]]:26 = #12
#define M4(x) ((x) && (x))
// CHECK-NEXT: File 6, [[@LINE+3]]:15 -> [[@LINE+3]]:27 = #0
// CHECK-NEXT: File 6, [[@LINE+2]]:16 -> [[@LINE+2]]:19 = #0
// CHECK-NEXT: File 6, [[@LINE+1]]:23 -> [[@LINE+1]]:26 = #14
#define M5(x) ((x) || (x))
// CHECK-NEXT: File 7, [[@LINE+1]]:15 -> [[@LINE+1]]:26 = #0
#define M6(x) ((x) + (x))
// CHECK-NEXT: File 8, [[@LINE+1]]:15 -> [[@LINE+1]]:18 = #0
#define M7(x) (x)

// Check for the expansion of M2 within M3.
// CHECK-NEXT: File 9, {{[0-9]+}}:15 -> {{[0-9]+}}:41 = (#0 + #8)
// CHECK-NEXT: File 9, {{[0-9]+}}:18 -> {{[0-9]+}}:31 = ((#0 + #8) + #9)
// CHECK-NEXT: File 9, {{[0-9]+}}:24 -> {{[0-9]+}}:25 = ((#0 + #8) + #9)
// CHECK: File 9, {{[0-9]+}}:27 -> {{[0-9]+}}:29 = #10
// CHECK-NEXT: File 9, {{[0-9]+}}:39 -> {{[0-9]+}}:40 = ((#0 + #8) + #9)

void func(int x) {
  if (x) {}
  M1;
  M2(!x);
  M2(x);
  M3(x);
  if (M4(x)) {}
  if (M5(x)) {}
  if (M6(x)) {}
  if (M7(x)) {}
}

int main(int argc, const char *argv[]) {
  func(0);
}
