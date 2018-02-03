// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

void testVLAwithSize(int s)
{
// CHECK-DAG: dbg.declare({{.*}} %vla_expr, metadata ![[VLAEXPR:[0-9]+]]
// CHECK-DAG: dbg.declare({{.*}} %vla, metadata ![[VAR:[0-9]+]]
// CHECK-DAG: ![[VLAEXPR]] = !DILocalVariable(name: "vla_expr"
// CHECK-DAG: ![[VAR]] = !DILocalVariable(name: "vla",{{.*}} line: [[@LINE+2]]
// CHECK-DAG: !DISubrange(count: ![[VLAEXPR]])
  int vla[s];
  int i;
  for (i = 0; i < s; i++) {
    vla[i] = i*i;
  }
}
