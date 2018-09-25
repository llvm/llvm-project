// RUN: %clang_cc1 -debug-info-kind=standalone -std=c++11 -triple x86_64-darwin -emit-llvm -o - %s | FileCheck %s

struct A {
  A(int, ...);
};
struct B : A {
  using A::A;
};

A::A(int i, ...) {}
// CHECK: define void @{{.*}}foo
// CHECK-NOT ret void
// CHECK: call void @llvm.dbg.declare
// CHECK-NOT ret void
// CHECK: call void @llvm.dbg.declare(metadata %{{.*}}** %{{[^,]+}},
// CHECK-SAME: metadata ![[THIS:[0-9]+]], metadata !DIExpression()), !dbg ![[LOC:[0-9]+]]
// CHECK: ret void, !dbg ![[NOINL:[0-9]+]]
// CHECK: ![[FOO:.*]] = distinct !DISubprogram(name: "foo"
// CHECK-DAG: ![[A:.*]] = distinct !DISubprogram(name: "A", linkageName: "_ZN1BCI11AEiz"
void foo() {
// CHECK-DAG: ![[LOC]] = !DILocation(line: 0, scope: ![[A]], inlinedAt: ![[INL:[0-9]+]])
// CHECK-DAG: ![[INL]] = !DILocation(line: [[@LINE+1]], scope: ![[FOO]])
  B b(0);
// CHECK: ![[NOINL]] = !DILocation(line: [[@LINE+1]], scope: !{{[0-9]+}})
}
