// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroscopes.cpp %s | FileCheck %s

#define starts_a_scope for (int i = 0; i < 2; ++i) {

#define ends_a_scope \
  x = x;             \
  }

#define some_code \
  x = x;          \
  if (x == 0) {   \
    x = 1;        \
  } else {        \
    x = 2;        \
  }               \
  if (true) {     \
    x = x;        \
  } else {        \
    x = x;        \
  }

#define starts_a_while while (x < 5)
#define simple_stmt ++x

#define macro_with_for          \
  x = 3;                        \
  for (int i = 0; i < x; ++i) { \
  }

#define macro_with_while \
  x = 4;                 \
  while (x < 5) {        \
    ++x;                 \
  }

// CHECK: main
// CHECK-NEXT: File 0, [[@LINE+1]]:12 -> {{[0-9]+}}:2 = #0
int main() {
  int x = 0;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:3 -> [[@LINE+2]]:17 = #0
  // CHECK-NEXT: File 0, [[@LINE+1]]:17 -> [[@LINE+7]]:15 = #1
  starts_a_scope
    x = x;
    // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:5 -> [[@LINE+1]]:14 = #1
    some_code
    x = x;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:15 = #1
  ends_a_scope

  // CHECK-NEXT: Expansion,File 0, [[@LINE+4]]:3 -> [[@LINE+4]]:17 = #0
  // CHECK-NEXT: File 0, [[@LINE+3]]:17 -> [[@LINE+5]]:15 = #4
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:5 -> [[@LINE+3]]:14 = #4
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:3 -> [[@LINE+3]]:15 = #4
  starts_a_scope
    some_code
  ends_a_scope

  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:3 -> [[@LINE+3]]:17 = #0
  // CHECK-NEXT: File 0, [[@LINE+2]]:17 -> [[@LINE+3]]:15 = #7
  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:3 -> [[@LINE+2]]:15 = #7
  starts_a_scope
  ends_a_scope

  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:3 -> [[@LINE+2]]:17 = #0
  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:5 -> [[@LINE+2]]:16 = #8
  starts_a_while
    simple_stmt;

  x = 0;
  // CHECK-NEXT: Expansion,File 0, [[@LINE+4]]:3 -> [[@LINE+4]]:17 = #0
  // CHECK-NEXT: File 0, [[@LINE+3]]:18 -> [[@LINE+5]]:15 = #9
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:5 -> [[@LINE+3]]:16 = #9
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:3 -> [[@LINE+3]]:15 = #9
  starts_a_while {
    simple_stmt;
  ends_a_scope

  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:17 = #0
  macro_with_for
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:19 = #0
  macro_with_while

  return 0;
}

// CHECK-NEXT: File 1, 3:24 -> 3:53 = #0
// CHECK-NEXT: File 1, 3:40 -> 3:45 = (#0 + #1)
// CHECK-NEXT: Branch,File 1, 3:40 -> 3:45 = #1, #0
// CHECK-NEXT: File 1, 3:47 -> 3:50 = #1
// CHECK-NEXT: File 1, 3:52 -> 3:53 = #1
// CHECK-NEXT: File 2, 10:3 -> 20:4 = #1
// CHECK-NEXT: File 2, 11:7 -> 11:13 = #1
// CHECK: File 2, 11:15 -> 13:4 = #2
// CHECK-NEXT: File 2, 13:10 -> 15:4 = (#1 - #2)
// CHECK-NEXT: File 2, 16:7 -> 16:11 = #1
// CHECK: File 2, 16:13 -> 18:4 = #3
// CHECK-NEXT: File 2, 18:10 -> 20:4 = (#1 - #3)
// CHECK-NEXT: File 3, 6:3 -> 7:4 = #1
// CHECK-NEXT: File 4, 3:24 -> 3:53 = #0
// CHECK-NEXT: File 4, 3:40 -> 3:45 = (#0 + #4)
// CHECK-NEXT: Branch,File 4, 3:40 -> 3:45 = #4, #0
// CHECK-NEXT: File 4, 3:47 -> 3:50 = #4
// CHECK-NEXT: File 4, 3:52 -> 3:53 = #4
// CHECK-NEXT: File 5, 10:3 -> 20:4 = #4
// CHECK-NEXT: File 5, 11:7 -> 11:13 = #4
// CHECK: File 5, 11:15 -> 13:4 = #5
// CHECK-NEXT: File 5, 13:10 -> 15:4 = (#4 - #5)
// CHECK-NEXT: File 5, 16:7 -> 16:11 = #4
// CHECK: File 5, 16:13 -> 18:4 = #6
// CHECK-NEXT: File 5, 18:10 -> 20:4 = (#4 - #6)
// CHECK-NEXT: File 6, 6:3 -> 7:4 = #4
// CHECK-NEXT: File 7, 3:24 -> 3:53 = #0
// CHECK-NEXT: File 7, 3:40 -> 3:45 = (#0 + #7)
// CHECK-NEXT: Branch,File 7, 3:40 -> 3:45 = #7, #0
// CHECK-NEXT: File 7, 3:47 -> 3:50 = #7
// CHECK-NEXT: File 7, 3:52 -> 3:53 = #7
// CHECK-NEXT: File 8, 6:3 -> 7:4 = #7
// CHECK-NEXT: File 9, 22:24 -> 22:37 = #0
// CHECK-NEXT: File 9, 22:31 -> 22:36 = (#0 + #8)
// CHECK-NEXT: Branch,File 9, 22:31 -> 22:36 = #8, #0
// CHECK-NEXT: File 10, 23:21 -> 23:24 = #8
// CHECK-NEXT: File 11, 22:24 -> 22:37 = #0
// CHECK-NEXT: File 11, 22:31 -> 22:36 = (#0 + #9)
// CHECK-NEXT: Branch,File 11, 22:31 -> 22:36 = #9, #0
// CHECK-NEXT: File 12, 23:21 -> 23:24 = #9
// CHECK-NEXT: File 13, 6:3 -> 7:4 = #9
// CHECK-NEXT: File 14, 26:3 -> 28:4 = #0
// CHECK-NEXT: File 14, 27:19 -> 27:24 = (#0 + #10)
// CHECK-NEXT: Branch,File 14, 27:19 -> 27:24 = #10, #0
// CHECK-NEXT: File 14, 27:26 -> 27:29 = #10
// CHECK-NEXT: File 14, 27:31 -> 28:4 = #10
// CHECK-NEXT: File 15, 31:3 -> 34:4 = #0
// CHECK-NEXT: File 15, 32:10 -> 32:15 = (#0 + #11)
// CHECK-NEXT: Branch,File 15, 32:10 -> 32:15 = #11, #0
// CHECK-NEXT: File 15, 32:17 -> 34:4 = #11
