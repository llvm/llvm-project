// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple %itanium_abi_triple -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

// CHECK-LABEL:       _Z19array_decompositioni:
// CHECK-NEXT:          File 0, [[@LINE+6]]:32 -> {{[0-9]+}}:2 = #0
// CHECK-NEXT:          File 0, [[@LINE+8]]:20 -> [[@LINE+8]]:25 = #0
// CHECK-NEXT:          Branch,File 0, [[@LINE+7]]:20 -> [[@LINE+7]]:25 = #1, (#0 - #1)
// CHECK-NEXT:          Gap,File 0, [[@LINE+6]]:27 -> [[@LINE+6]]:28 = #1
// CHECK-NEXT:          File 0, [[@LINE+5]]:28 -> [[@LINE+5]]:29 = #1
// CHECK-NEXT:          File 0, [[@LINE+4]]:32 -> [[@LINE+4]]:33 = (#0 - #1)
int array_decomposition(int i) {
  int a[] = {1, 2, 3};
  int b[] = {4, 5, 6};
  auto [x, y, z] = i > 0 ? a : b;
  return x + y + z;
}
