// RUN: %clang -target x86_64-unknown-unknown -fverbose-asm -g -O0 -S -emit-llvm %s -o - | FileCheck %s


void f(int m) {
  int x[3][m];
}

// CHECK: !DICompositeType(tag: DW_TAG_array_type,
// CHECK-NOT:                               size:
// CHECK-SAME:                              align: 32
// CHECK-SAME:                              elements: [[ELEM_TYPE:![0-9]+]]
// CHECK: [[ELEM_TYPE]] = !{[[SUB1:.*]], [[SUB2:.*]]}
// CHECK: [[SUB1]] = !DISubrange(count: 3)
// CHECK: [[SUB2]] = !DISubrange(count: -1)
