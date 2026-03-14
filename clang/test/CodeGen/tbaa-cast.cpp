// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s \
// RUN:     -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s \
// RUN:     -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECK,NEW-PATH
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

// OLD-PATH-DAG: [[TAG_V_n]] = !{[[TYPE_V:!.*]], [[TYPE_int:!.*]], i64 0}
// OLD-PATH-DAG: [[TYPE_V]] = !{!"_ZTS1V", !{{.*}}, i64 0}
// OLD-PATH-DAG: [[TYPE_int]] = !{!"int", !{{.*}}, i64 0}
// NEW-PATH-DAG: [[TAG_V_n]] = !{[[TYPE_V:!.*]], [[TYPE_int:!.*]], i64 0, i64 4}
// NEW-PATH-DAG: [[TYPE_V]] = !{[[TYPE_char:!.*]], i64 4, !"_ZTS1V", [[TYPE_int]], i64 0, i64 4}
// NEW-PATH-DAG: [[TYPE_int]] = !{[[TYPE_char]], i64 4, !"int"}
