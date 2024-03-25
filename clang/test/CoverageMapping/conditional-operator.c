// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

// CHECK-LABEL:       binary_conditional:
// CHECK-NEXT:          File 0, [[@LINE+4]]:31 -> {{[0-9]+}}:2 = #0
// CHECK-NEXT:          File 0, [[@LINE+4]]:7 -> [[@LINE+4]]:8 = #0
// CHECK-NEXT:          Branch,File 0, [[@LINE+3]]:7 -> [[@LINE+3]]:8 = #1, (#0 - #1)
// CHECK-NEXT:          File 0, [[@LINE+2]]:13 -> [[@LINE+2]]:14 = (#0 - #1)
int binary_conditional(int x) {
  x = x ? : 4;
  int y = x;
  return y;
}

// CHECK-LABEL:       ternary_conditional:
// CHECK-NEXT:          File 0, [[@LINE+6]]:32 -> {{[0-9]+}}:2 = #0
// CHECK-NEXT:          File 0, [[@LINE+6]]:7 -> [[@LINE+6]]:8 = #0
// CHECK-NEXT:          Branch,File 0, [[@LINE+5]]:7 -> [[@LINE+5]]:8 = #1, (#0 - #1)
// CHECK-NEXT:          Gap,File 0, [[@LINE+4]]:10 -> [[@LINE+4]]:11 = #1
// CHECK-NEXT:          File 0, [[@LINE+3]]:11 -> [[@LINE+3]]:12 = #1
// CHECK-NEXT:          File 0, [[@LINE+2]]:15 -> [[@LINE+2]]:16 = (#0 - #1)
int ternary_conditional(int x) {
  x = x ? x : 4;
  int y = x;
  return y;
}
