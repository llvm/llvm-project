// RUN: %clang_cc1 -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macro-expressions.cpp -w %s | FileCheck %s

#define EXPR(x) (x)
#define NEXPR(x) (!x)
#define DECL(T, x) T x
#define ASSIGN(x, y) x = y
#define LT(x, y) x < y
#define INC(x) ++x
#define ARR(T, x, y, z) (T[3]){x, y, z}

#define PRI_64_LENGTH_MODIFIER "ll"
#define PRIo64 PRI_64_LENGTH_MODIFIER "o"
#define PRIu64 PRI_64_LENGTH_MODIFIER "u"

#define STMT(s) s

void fn1() {
  STMT(if (1));
  STMT(while (1));
  STMT(for (;;));
  STMT(if) (1);
  STMT(while) (1);
  STMT(for) (;;);
  if (1)
    STMT(if (1)
        STMT(if (1)));
  if (1)
    STMT(if (1)) 0;
  if (1)
    STMT(while (1)) 0;
  if (1)
    STMT(for (;;)) 0;
  while (1)
    STMT(if (1)) 0;
  while (1)
    STMT(while (1)) 0;
  while (1)
    STMT(for (;;)) 0;
  for (;;)
    STMT(if (1)) 0;
  for (;;)
    STMT(while (1)) 0;
  for (;;)
    STMT(for (;;)) 0;
}

void STMT(fn2()) {
}

void STMT(fn3)() {
}

// CHECK: foo
// CHECK-NEXT: File 0, [[@LINE+1]]:17 -> {{[0-9]+}}:2 = #0
void foo(int i) {
  // CHECK-NEXT: File 0, [[@LINE+2]]:7 -> [[@LINE+2]]:8 = #0
  // CHECK-NEXT: File 0, [[@LINE+1]]:10 -> [[@LINE+1]]:12 = #1
  if (0) {}

  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:7 -> [[@LINE+2]]:11 = #0
  // CHECK-NEXT: File 0, [[@LINE+1]]:16 -> [[@LINE+1]]:18 = #2
  if (EXPR(i)) {}
  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:9 -> [[@LINE+2]]:14 = (#0 + #3)
  // CHECK-NEXT: File 0, [[@LINE+1]]:20 -> [[@LINE+1]]:22 = #3
  for (;NEXPR(i);) {}
  // CHECK-NEXT: Expansion,File 0, [[@LINE+4]]:8 -> [[@LINE+4]]:14 = #0
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:33 -> [[@LINE+3]]:35 = (#0 + #4)
  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:43 -> [[@LINE+2]]:46 = #4
  // CHECK-NEXT: File 0, [[@LINE+1]]:51 -> [[@LINE+1]]:53 = #4
  for (ASSIGN(DECL(int, j), 0); LT(j, i); INC(j)) {}
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:9 = #0
  ASSIGN(DECL(int, k), 0);
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:10 -> [[@LINE+3]]:12 = (#0 + #5)
  // CHECK-NEXT: File 0, [[@LINE+2]]:20 -> [[@LINE+2]]:31 = #5
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:22 -> [[@LINE+1]]:25 = #5
  while (LT(k, i)) { INC(k); }
  // CHECK-NEXT: File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:8 = (#0 + #6)
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:16 -> [[@LINE+1]]:21 = (#0 + #6)
  do {} while (NEXPR(i));
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:8 -> [[@LINE+3]]:12 = #0
  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:23 -> [[@LINE+2]]:26 = #0
  // CHECK-NEXT: File 0, [[@LINE+1]]:42 -> [[@LINE+1]]:44 = #7
  for (DECL(int, j) : ARR(int, 1, 2, 3)) {}

  // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:14 -> [[@LINE+2]]:20 = #0
  // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:23 -> [[@LINE+1]]:29 = #0
  (void)(i ? PRIo64 : PRIu64);

  // CHECK-NEXT: File 0, [[@LINE+5]]:14 -> [[@LINE+5]]:15 = #9
  // CHECK-NEXT: Expansion,File 0, [[@LINE+4]]:18 -> [[@LINE+4]]:22 = (#0 - #9)
  // CHECK-NEXT: File 0, [[@LINE+3]]:22 -> [[@LINE+3]]:33 = (#0 - #9)
  // CHECK-NEXT: File 0, [[@LINE+2]]:28 -> [[@LINE+2]]:29 = #10
  // CHECK-NEXT: File 0, [[@LINE+1]]:32 -> [[@LINE+1]]:33 = ((#0 - #9) - #10)
  (void)(i ? i : EXPR(i) ? i : 0);
  // CHECK-NEXT: Expansion,File 0, [[@LINE+3]]:15 -> [[@LINE+3]]:19 = (#0 - #11)
  // CHECK-NEXT: File 0, [[@LINE+2]]:19 -> [[@LINE+2]]:27 = (#0 - #11)
  // CHECK-NEXT: File 0, [[@LINE+1]]:26 -> [[@LINE+1]]:27 = ((#0 - #11) - #12)
  (void)(i ?: EXPR(i) ?: 0);
}

// CHECK-NEXT: File {{[0-9]+}}, 3:17 -> 3:20 = #0
// CHECK-NEXT: File {{[0-9]+}}, 4:18 -> 4:22 = (#0 + #3)
// CHECK-NEXT: File {{[0-9]+}}, 6:22 -> 6:27 = #0
// CHECK-NEXT: File {{[0-9]+}}, 8:16 -> 8:19 = #4
// CHECK-NEXT: File {{[0-9]+}}, 7:18 -> 7:23 = (#0 + #4)
// CHECK-NEXT: File {{[0-9]+}}, 6:22 -> 6:27 = #0
// CHECK-NEXT: File {{[0-9]+}}, 8:16 -> 8:19 = #5
// CHECK-NEXT: File {{[0-9]+}}, 7:18 -> 7:23 = (#0 + #5)
// CHECK-NEXT: File {{[0-9]+}}, 4:18 -> 4:22 = (#0 + #6)
// CHECK-NEXT: File {{[0-9]+}}, 5:20 -> 5:23 = #0
// CHECK-NEXT: File {{[0-9]+}}, 9:25 -> 9:40 = #0
// CHECK-NEXT: File {{[0-9]+}}, 12:16 -> 12:42 = #0
// CHECK-NEXT: Expansion,File {{[0-9]+}}, 12:16 -> 12:38 = #8
// CHECK-NEXT: File {{[0-9]+}}, 12:38 -> 12:42 = #8
// CHECK-NEXT: File {{[0-9]+}}, 13:16 -> 13:42 = #0
// CHECK-NEXT: Expansion,File {{[0-9]+}}, 13:16 -> 13:38 = (#0 - #8)
// CHECK-NEXT: File {{[0-9]+}}, 13:38 -> 13:42 = (#0 - #8)
// CHECK-NEXT: File {{[0-9]+}}, 3:17 -> 3:20 = (#0 - #9)
// CHECK-NEXT: File {{[0-9]+}}, 3:17 -> 3:20 = (#0 - #11)
// CHECK-NEXT: File {{[0-9]+}}, 11:32 -> 11:36 = #8
// CHECK-NEXT: File {{[0-9]+}}, 11:32 -> 11:36 = (#0 - #8)

// CHECK-NOT: File {{[0-9]+}},
// CHECK: main

int main(int argc, const char *argv[]) {
  foo(10);
}
