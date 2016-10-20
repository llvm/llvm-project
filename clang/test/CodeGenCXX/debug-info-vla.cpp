// RUN: %clang -target x86_64-unknown-unknown -fverbose-asm -g -O0 -S -emit-llvm %s -o - -std=c++11 | FileCheck %s


void f(int m) {
  int x[3][m];
}

int (*fp)(int[][*]) = nullptr;

// CHECK: !DICompositeType(tag: DW_TAG_array_type,
// CHECK-NOT:                               size:
// CHECK-SAME:                              elements: [[ELEM_TYPE:![0-9]+]]
// CHECK: [[ELEM_TYPE]] = !{[[NOCOUNT:.*]]}
// CHECK: [[NOCOUNT]] = !DISubrange(count: -1)
//
// CHECK: !DICompositeType(tag: DW_TAG_array_type,
// CHECK-NOT:                               size:
// CHECK-SAME:                              elements: [[ELEM_TYPE:![0-9]+]]
// CHECK: [[ELEM_TYPE]] = !{[[THREE:.*]], [[NOCOUNT]]}
// CHECK: [[THREE]] = !DISubrange(count: 3)
