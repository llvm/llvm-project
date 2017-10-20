// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s \
// RUN:     -emit-llvm -o - | FileCheck %s
//
// Check that we generate correct TBAA information for accesses to array
// elements.

struct A { int i; };
struct B { A a[1]; };

int foo(B *b) {
// CHECK-LABEL: _Z3fooP1B
// CHECK: load i32, {{.*}}, !tbaa [[TAG_A_i:!.*]]
  return b->a->i;
}

// CHECK-DAG: [[TAG_A_i]] = !{[[TYPE_A:!.*]], [[TYPE_int:!.*]], i64 0}
// CHECK-DAG: [[TYPE_A]] = !{!"_ZTS1A", !{{.*}}, i64 0}
// CHECK-DAG: [[TYPE_int]] = !{!"int", !{{.*}}, i64 0}
