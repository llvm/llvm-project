// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 -no-struct-path-tbaa -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 -disable-llvm-passes -o - %s | FileCheck %s -check-prefix=PATH

// Types with the may_alias attribute should be considered equivalent
// to char for aliasing.

typedef int __attribute__((may_alias)) aliasing_int;

void test0(aliasing_int *ai, int *i)
{
// CHECK: store i32 0, i32* %{{.*}}, !tbaa [[TAG_CHAR:!.*]]
// PATH: store i32 0, i32* %{{.*}}, !tbaa [[TAG_CHAR:!.*]]
  *ai = 0;
// CHECK: store i32 1, i32* %{{.*}}, !tbaa [[TAG_INT:!.*]]
// PATH: store i32 1, i32* %{{.*}}, !tbaa [[TAG_INT:!.*]]
  *i = 1;
}

// PR9307
struct Test1 { int x; };
struct Test1MA { int x; } __attribute__((may_alias));
void test1(struct Test1MA *p1, struct Test1 *p2) {
  // CHECK: store i32 2, i32* {{%.*}}, !tbaa [[TAG_CHAR]]
  // PATH: store i32 2, i32* {{%.*}}, !tbaa [[TAG_CHAR]]
  p1->x = 2;
  // CHECK: store i32 3, i32* {{%.*}}, !tbaa [[TAG_INT]]
  // PATH: store i32 3, i32* {{%.*}}, !tbaa [[TAG_test1_x:!.*]]
  p2->x = 3;
}
// CHECK:  !"any pointer", [[TYPE_CHAR:!.*]],
// CHECK: [[TYPE_CHAR]] = !{!"omnipotent char", [[TAG_CXX_TBAA:!.*]],
// CHECK: [[TAG_CXX_TBAA]] = !{!"Simple C/C++ TBAA"}
// CHECK: [[TAG_CHAR]] = !{[[TYPE_CHAR]], [[TYPE_CHAR]], i64 0}
// CHECK: [[TAG_INT]] = !{[[TYPE_INT:!.*]], [[TYPE_INT]], i64 0}
// CHECK: [[TYPE_INT]] = !{!"int", [[TYPE_CHAR]]

// PATH: [[TYPE_CHAR:!.*]] = !{!"omnipotent char", !{{.*}}
// PATH: [[TAG_CHAR]] = !{[[TYPE_CHAR]], [[TYPE_CHAR]], i64 0}
// PATH: [[TAG_INT]] = !{[[TYPE_INT:!.*]], [[TYPE_INT]], i64 0}
// PATH: [[TYPE_INT]] = !{!"int", [[TYPE_CHAR]]
// PATH: [[TAG_test1_x]] = !{[[TYPE_test1:!.*]], [[TYPE_INT]], i64 0}
// PATH: [[TYPE_test1]] = !{!"Test1", [[TYPE_INT]], i64 0}
