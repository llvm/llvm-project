// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s \
// RUN:     -emit-llvm -o - | FileCheck %s
//
// Check that we generate correct TBAA information for lvalues constructed
// with use of casts.

struct V {
  unsigned n;
};

struct S {
  char bytes[4];
};

void foo(S *p) {
// CHECK-LABEL: _Z3fooP1S
// CHECK: store i32 5, {{.*}}, !tbaa [[TAG_V_n:!.*]]
  ((V*)p->bytes)->n = 5;
}

// CHECK-DAG: [[TAG_V_n]] = !{[[TYPE_V:!.*]], [[TYPE_int:!.*]], i64 0}
// CHECK-DAG: [[TYPE_V]] = !{!"_ZTS1V", !{{.*}}, i64 0}
// CHECK-DAG: [[TYPE_int]] = !{!"int", !{{.*}}, i64 0}
