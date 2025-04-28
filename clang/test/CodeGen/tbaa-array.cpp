// RUN: %clang_cc1 -triple x86_64-linux -O1 %s \
// RUN:     -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -O1 %s \
// RUN:     -new-struct-path-tbaa -emit-llvm -o - | \
// RUN:     FileCheck -check-prefix=CHECK-NEW %s
//
// Check that we generate correct TBAA information for accesses to array
// elements.

struct A { int i; };
struct B { A a[1]; };
struct C { int i; int x[3]; };
struct D { int n; int arr[]; }; // flexible array member
extern int AA[];                // incomplete array type

int foo(B *b) {
// CHECK-LABEL: _Z3fooP1B
// CHECK: load i32, {{.*}}, !tbaa [[TAG_A_i:!.*]]
// CHECK-NEW-LABEL: _Z3fooP1B
// CHECK-NEW: load i32, {{.*}}, !tbaa [[TAG_A_i:!.*]]
  return b->a->i;
}

// Check that members of array types are represented correctly.
int bar(C *c) {
// CHECK-NEW-LABEL: _Z3barP1C
// CHECK-NEW: load i32, {{.*}}, !tbaa [[TAG_C_i:!.*]]
  return c->i;
}

int bar2(C *c) {
// CHECK-NEW-LABEL: _Z4bar2P1C
// CHECK-NEW: load i32, {{.*}}, !tbaa [[TAG_C_x:!.*]]
  return c->x[2];
}

int bar3(C *c, int j) {
// CHECK-NEW-LABEL: _Z4bar3P1Ci
// CHECK-NEW: load i32, {{.*}}, !tbaa [[TAG_C_x]]
  return c->x[j];
}

int bar4(D *d) {
// CHECK-NEW-LABEL: _Z4bar4P1D
// CHECK-NEW: load i32, {{.*}}, !tbaa [[TAG_int:!.*]]
// CHECK-NEW: load i32, {{.*}}, !tbaa [[TAG_int:!.*]]
    return d->arr[d->n];
}

int bar5(int j) {
// CHECK-NEW-LABEL: _Z4bar5i
// CHECK-NEW: load i32, {{.*}}, !tbaa [[TAG_int:!.*]]
// CHECK-NEW: load i32, {{.*}}, !tbaa [[TAG_int:!.*]]
    return AA[2] + AA[j];
}

// CHECK-DAG: [[TAG_A_i]] = !{[[TYPE_A:!.*]], [[TYPE_int:!.*]], i64 0}
// CHECK-DAG: [[TYPE_A]] = !{!"_ZTS1A", !{{.*}}, i64 0}
// CHECK-DAG: [[TYPE_int]] = !{!"int", !{{.*}}, i64 0}

// CHECK-NEW-DAG: [[TYPE_char:!.*]] = !{{{.*}}, i64 1, !"omnipotent char"}
// CHECK-NEW-DAG: [[TYPE_int:!.*]] = !{[[TYPE_char]], i64 4, !"int"}
// CHECK-NEW-DAG: [[TAG_int]] = !{[[TYPE_int]], [[TYPE_int]], i64 0, i64 4}
// CHECK-NEW-DAG: [[TYPE_A:!.*]] = !{[[TYPE_char]], i64 4, !"_ZTS1A", [[TYPE_int]], i64 0, i64 4}
// CHECK-NEW-DAG: [[TAG_A_i]] = !{[[TYPE_A]], [[TYPE_int]], i64 0, i64 4}
// CHECK-NEW-DAG: [[TYPE_C:!.*]] = !{[[TYPE_char]], i64 16, !"_ZTS1C", [[TYPE_int]], i64 0, i64 4, [[TYPE_int]], i64 4, i64 12}
// CHECK-NEW-DAG: [[TAG_C_i]] = !{[[TYPE_C:!.*]], [[TYPE_int:!.*]], i64 0, i64 4}
// CHECK-NEW-DAG: [[TAG_C_x]] = !{[[TYPE_C:!.*]], [[TYPE_int:!.*]], i64 4, i64 4}
