// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 -disable-llvm-optzns -o %t %s
// RUN: FileCheck < %t %s

// Types with the may_alias attribute should be considered equivalent
// to char for aliasing.

typedef int __attribute__((may_alias)) aliasing_int;

void test0(aliasing_int *ai, int *i)
{
// CHECK: store i32 0, i32* %{{.*}}, !tbaa !1
  *ai = 0;
// CHECK: store i32 1, i32* %{{.*}}, !tbaa !3
  *i = 1;
}

// PR9307
struct Test1 { int x; };
struct Test1MA { int x; } __attribute__((may_alias));
void test1(struct Test1MA *p1, struct Test1 *p2) {
  // CHECK: store i32 2, i32* {{%.*}}, !tbaa !1
  p1->x = 2;
  // CHECK: store i32 3, i32* {{%.*}}, !tbaa !3
  p2->x = 3;
}

// CHECK: !0 = metadata !{metadata !"any pointer", metadata !1}
// CHECK: !1 = metadata !{metadata !"omnipotent char", metadata !2}
// CHECK: !2 = metadata !{metadata !"Simple C/C++ TBAA"}
// CHECK: !3 = metadata !{metadata !"int", metadata !1}
