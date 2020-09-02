// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name loopmacro.c %s | FileCheck %s

// CHECK: main
// CHECK-NEXT: File 0, {{[0-9]+}}:12 -> {{[0-9]+}}:2 = #0
// CHECK-NEXT: File 0, {{[0-9]+}}:6 -> {{[0-9]+}}:4 = (#0 + #1)
// CHECK-NEXT: Expansion,File 0, {{[0-9]+}}:7 -> {{[0-9]+}}:20 = (#0 + #1)
// CHECK-NEXT: File 0, {{[0-9]+}}:12 -> {{[0-9]+}}:30 = (#0 + #1)

// CHECK-NEXT: File 1, [[@LINE+4]]:4 -> [[@LINE+6]]:23 = (#0 + #1)
// CHECK-NEXT: Expansion,File 1, [[@LINE+3]]:5 -> [[@LINE+3]]:16 = (#0 + #1)
// CHECK-NEXT: Expansion,File 1, [[@LINE+3]]:16 -> [[@LINE+3]]:21 = (#0 + #1)
#define INSERT_STRING(s, match_head) \
   (UPDATE_HASH(ins_h, window[(s) + MIN_MATCH-1]), \
    prev[(s) & WMASK] = match_head = head[ins_h], \
    head[ins_h] = (s))
// CHECK-NEXT: File 2, [[@LINE+3]]:26 -> [[@LINE+3]]:66 = (#0 + #1)
// CHECK-NEXT: Expansion,File 2, [[@LINE+2]]:38 -> [[@LINE+2]]:45 = (#0 + #1)
// CHECK-NEXT: Expansion,File 2, [[@LINE+1]]:56 -> [[@LINE+1]]:65 = (#0 + #1)
#define UPDATE_HASH(h,c) (h = (((h)<<H_SHIFT) ^ (c)) & HASH_MASK)
// CHECK-NEXT: File 3, [[@LINE+1]]:15 -> [[@LINE+1]]:21 = (#0 + #1)
#define WMASK 0xFFFF
// CHECK-NEXT: File 4, [[@LINE+4]]:18 -> [[@LINE+4]]:53 = (#0 + #1)
// CHECK-NEXT: Expansion,File 4, [[@LINE+3]]:20 -> [[@LINE+3]]:29 = (#0 + #1)
// CHECK-NEXT: Expansion,File 4, [[@LINE+2]]:30 -> [[@LINE+2]]:39 = (#0 + #1)
// CHECK-NEXT: Expansion,File 4, [[@LINE+1]]:43 -> [[@LINE+1]]:52 = (#0 + #1)
#define H_SHIFT  ((HASH_BITS+MIN_MATCH-1)/MIN_MATCH)
// CHECK-NEXT: File 5, [[@LINE+1]]:19 -> [[@LINE+1]]:25 = (#0 + #1)
#define HASH_MASK 0xFFFF
// CHECK-NEXT: File 6, [[@LINE+1]]:20 -> [[@LINE+1]]:22 = (#0 + #1)
#define HASH_BITS  15
// CHECK-NEXT: File 7, [[@LINE+2]]:20 -> [[@LINE+2]]:21 = (#0 + #1)
// CHECK-NEXT: File 8, [[@LINE+1]]:20 -> [[@LINE+1]]:21 = (#0 + #1)
#define MIN_MATCH  3

int main() {
  int strstart = 0;
  int hash_head = 2;
  int prev_length = 5;
  int ins_h = 1;
  int prev[32<<10] = { 0 };
  int head[32<<10] = { 0 };
  int window[1024] = { 0 };
  do {
      strstart++;
      INSERT_STRING(strstart, hash_head);
  } while (--prev_length != 0);
}
