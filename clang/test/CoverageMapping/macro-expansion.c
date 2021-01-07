// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macro-expansion.c %s | FileCheck %s

// CHECK: func
// CHECK:      File 1, [[@LINE+7]]:12 -> [[@LINE+7]]:38 = #0
// CHECK-NEXT: File 1, [[@LINE+6]]:15 -> [[@LINE+6]]:28 = (#0 + #2)
// CHECK-NEXT: File 1, [[@LINE+5]]:21 -> [[@LINE+5]]:22 = (#0 + #2)
// CHECK: Branch,File 1, [[@LINE+4]]:21 -> [[@LINE+4]]:22 = 0, 0
// CHECK-NEXT: File 1, [[@LINE+3]]:24 -> [[@LINE+3]]:26 = #3
// CHECK-NEXT: File 1, [[@LINE+2]]:36 -> [[@LINE+2]]:37 = (#0 + #2)
// CHECK-NEXT: Branch,File 1, [[@LINE+1]]:36 -> [[@LINE+1]]:37 = 0, 0
#define M1 do { if (0) {} } while (0)
// CHECK-NEXT: File 2, [[@LINE+12]]:15 -> [[@LINE+12]]:41 = #0
// CHECK-NEXT: File 2, [[@LINE+11]]:18 -> [[@LINE+11]]:31 = (#0 + #4)
// CHECK-NEXT: File 2, [[@LINE+10]]:24 -> [[@LINE+10]]:25 = (#0 + #4)
// CHECK: File 2, [[@LINE+9]]:27 -> [[@LINE+9]]:29 = #5
// CHECK-NEXT: File 2, [[@LINE+8]]:39 -> [[@LINE+8]]:40 = (#0 + #4)
// CHECK-NEXT: Branch,File 2, [[@LINE+7]]:39 -> [[@LINE+7]]:40 = 0, 0
// CHECK-NEXT: File 3, [[@LINE+6]]:15 -> [[@LINE+6]]:41 = #0
// CHECK-NEXT: File 3, [[@LINE+5]]:18 -> [[@LINE+5]]:31 = (#0 + #6)
// CHECK-NEXT: File 3, [[@LINE+4]]:24 -> [[@LINE+4]]:25 = (#0 + #6)
// CHECK: File 3, [[@LINE+3]]:27 -> [[@LINE+3]]:29 = #7
// CHECK-NEXT: File 3, [[@LINE+2]]:39 -> [[@LINE+2]]:40 = (#0 + #6)
// CHECK-NEXT: Branch,File 3, [[@LINE+1]]:39 -> [[@LINE+1]]:40 = 0, 0
#define M2(x) do { if (x) {} } while (0)
// CHECK-NEXT: File 4, [[@LINE+5]]:15 -> [[@LINE+5]]:38 = #0
// CHECK-NEXT: File 4, [[@LINE+4]]:18 -> [[@LINE+4]]:28 = (#0 + #8)
// CHECK-NEXT: Expansion,File 4, [[@LINE+3]]:20 -> [[@LINE+3]]:22 = (#0 + #8)
// CHECK-NEXT: File 4, [[@LINE+2]]:36 -> [[@LINE+2]]:37 = (#0 + #8)
// CHECK-NEXT: Branch,File 4, [[@LINE+1]]:36 -> [[@LINE+1]]:37 = 0, 0
#define M3(x) do { M2(x); } while (0)
// CHECK-NEXT: File 5, [[@LINE+4]]:15 -> [[@LINE+4]]:27 = #0
// CHECK-NEXT: File 5, [[@LINE+3]]:16 -> [[@LINE+3]]:19 = #0
// CHECK-NEXT: Branch,File 5, [[@LINE+2]]:16 -> [[@LINE+2]]:19 = #12, (#0 - #12)
// CHECK-NEXT: File 5, [[@LINE+1]]:23 -> [[@LINE+1]]:26 = #12
#define M4(x) ((x) && (x))
// CHECK-NEXT: Branch,File 5, [[@LINE-1]]:23 -> [[@LINE-1]]:26 = #13, (#12 - #13)
// CHECK-NEXT: File 6, [[@LINE+4]]:15 -> [[@LINE+4]]:27 = #0
// CHECK-NEXT: File 6, [[@LINE+3]]:16 -> [[@LINE+3]]:19 = #0
// CHECK-NEXT: Branch,File 6, [[@LINE+2]]:16 -> [[@LINE+2]]:19 = (#0 - #15), #15
// CHECK-NEXT: File 6, [[@LINE+1]]:23 -> [[@LINE+1]]:26 = #15
#define M5(x) ((x) || (x))
// CHECK-NEXT: Branch,File 6, [[@LINE-1]]:23 -> [[@LINE-1]]:26 = (#15 - #16), #16
// CHECK-NEXT: File 7, [[@LINE+1]]:15 -> [[@LINE+1]]:26 = #0
#define M6(x) ((x) + (x))
// CHECK-NEXT: Branch,File 7, [[@LINE-1]]:15 -> [[@LINE-1]]:26 = #17, (#0 - #17)
// CHECK-NEXT: File 8, [[@LINE+1]]:15 -> [[@LINE+1]]:18 = #0
#define M7(x) (x)

// Check for the expansion of M2 within M3.
// CHECK-NEXT: Branch,File 8, [[@LINE-3]]:15 -> [[@LINE-3]]:18 = #18, (#0 - #18)
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
