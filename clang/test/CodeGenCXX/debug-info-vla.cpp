// REQUIRES: rdar42833777
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
<<<<<<< HEAD
// CHECK: [[VAR:![0-9]+]] = !DILocalVariable(name: "__vla_expr", {{.*}}flags: DIFlagArtificial
||||||| 87815378b0e... Recommit rL323952: [DebugInfo] Enable debug information for C99 VLA types.
// CHECK: [[VAR:![0-9]+]] = !DILocalVariable(name: "vla_expr"
=======
>>>>>>> parent of 87815378b0e... Recommit rL323952: [DebugInfo] Enable debug information for C99 VLA types.
// CHECK: !DICompositeType(tag: DW_TAG_array_type,
// CHECK-NOT:                               size:
// CHECK-SAME:                              elements: [[ELEM_TYPE:![0-9]+]]
// CHECK: [[ELEM_TYPE]] = !{[[THREE:.*]], [[NOCOUNT]]}
// CHECK: [[THREE]] = !DISubrange(count: 3)
