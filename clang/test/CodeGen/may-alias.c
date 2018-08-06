// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 \
// RUN:     -no-struct-path-tbaa -disable-llvm-passes -o - %s | \
// RUN:     FileCheck -allow-deprecated-dag-overlap %s -check-prefixes=CHECK,SCALAR
// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 \
// RUN:     -disable-llvm-passes -o - %s | \
// RUN:     FileCheck -allow-deprecated-dag-overlap %s -check-prefixes=CHECK,OLD-PATH
// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 \
// RUN:     -new-struct-path-tbaa -disable-llvm-passes -o - %s | \
// RUN:     FileCheck -allow-deprecated-dag-overlap %s -check-prefixes=CHECK,NEW-PATH

// Types with the may_alias attribute should be considered equivalent
// to char for aliasing.

typedef int __attribute__((may_alias)) aliasing_int;

void test0(aliasing_int *ai, int *i) {
// CHECK-LABEL: test0
// CHECK: store i32 0, {{.*}}, !tbaa [[TAG_alias_int:!.*]]
  *ai = 0;

// CHECK: store i32 1, {{.*}}, !tbaa [[TAG_int:!.*]]
  *i = 1;
}

// PR9307
struct Test1 { int x; };
struct Test1MA { int x; } __attribute__((may_alias));
void test1(struct Test1MA *p1, struct Test1 *p2) {
// CHECK-LABEL: test1
// CHECK: store i32 2, {{.*}}, !tbaa [[TAG_alias_test1_x:!.*]]
  p1->x = 2;

// CHECK: store i32 3, {{.*}}, !tbaa [[TAG_test1_x:!.*]]
  p2->x = 3;
}

// SCALAR-DAG: [[ROOT:!.*]] = !{!"Simple C/C++ TBAA"}
// SCALAR-DAG: [[TYPE_char:!.*]] = !{!"omnipotent char", [[ROOT]], i64 0}
// SCALAR-DAG: [[TAG_alias_int]] = !{[[TYPE_char]], [[TYPE_char]], i64 0}
// SCALAR-DAG: [[TAG_alias_test1_x]] = !{[[TYPE_char]], [[TYPE_char]], i64 0}
// SCALAR-DAG: [[TYPE_int:!.*]] = !{!"int", [[TYPE_char]], i64 0}
// SCALAR-DAG: [[TAG_int]] = !{[[TYPE_int]], [[TYPE_int]], i64 0}
// SCALAR-DAG: [[TAG_test1_x]] = !{[[TYPE_int]], [[TYPE_int]], i64 0}

// OLD-PATH-DAG: [[ROOT:!.*]] = !{!"Simple C/C++ TBAA"}
// OLD-PATH-DAG: [[TYPE_char:!.*]] = !{!"omnipotent char", [[ROOT]], i64 0}
// OLD-PATH-DAG: [[TAG_alias_int]] = !{[[TYPE_char]], [[TYPE_char]], i64 0}
// OLD-PATH-DAG: [[TAG_alias_test1_x]] = !{[[TYPE_char]], [[TYPE_char]], i64 0}
// OLD-PATH-DAG: [[TYPE_int:!.*]] = !{!"int", [[TYPE_char]], i64 0}
// OLD-PATH-DAG: [[TAG_int]] = !{[[TYPE_int]], [[TYPE_int]], i64 0}
// OLD-PATH-DAG: [[TYPE_test1:!.*]] = !{!"Test1", [[TYPE_int]], i64 0}
// OLD-PATH-DAG: [[TAG_test1_x]] = !{[[TYPE_test1]], [[TYPE_int]], i64 0}

// NEW-PATH-DAG: [[ROOT:!.*]] = !{!"Simple C/C++ TBAA"}
// NEW-PATH-DAG: [[TYPE_char:!.*]] = !{[[ROOT]], i64 1, !"omnipotent char"}
// NEW-PATH-DAG: [[TAG_alias_int]] = !{[[TYPE_char]], [[TYPE_char]], i64 0, i64 0}
// NEW-PATH-DAG: [[TAG_alias_test1_x]] = !{[[TYPE_char]], [[TYPE_char]], i64 0, i64 0}
// NEW-PATH-DAG: [[TYPE_int:!.*]] = !{[[TYPE_char]], i64 4, !"int"}
// NEW-PATH-DAG: [[TAG_int]] = !{[[TYPE_int]], [[TYPE_int]], i64 0, i64 4}
// NEW-PATH-DAG: [[TYPE_test1:!.*]] = !{[[TYPE_char]], i64 4, !"Test1", [[TYPE_int]], i64 0, i64 4}
// NEW-PATH-DAG: [[TAG_test1_x]] = !{[[TYPE_test1]], [[TYPE_int]], i64 0, i64 4}
