// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -std=c++11 -triple x86_64-unknown-unknown %s -o - | FileCheck %s


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
// CHECK: [[VAR:![0-9]+]] = !DILocalVariable(name: "__vla_expr0", {{.*}}flags: DIFlagArtificial
// CHECK: !DICompositeType(tag: DW_TAG_array_type,
// CHECK-NOT:                               size:
// CHECK-SAME:                              elements: [[ELEM_TYPE:![0-9]+]]
// CHECK: [[ELEM_TYPE]] = !{[[THREE:.*]], [[VARRANGE:![0-9]+]]}
// CHECK: [[THREE]] = !DISubrange(count: 3)
// CHECK: [[VARRANGE]] = !DISubrange(count: [[VAR]])
