// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

void testVLAwithSize(int s)
{
// CHECK-DAG: #dbg_declare({{.*}} %__vla_expr0, ![[VLAEXPR:[0-9]+]]
// CHECK-DAG: #dbg_declare({{.*}} %vla, ![[VAR:[0-9]+]]
// CHECK-DAG: ![[VLAEXPR]] = !DILocalVariable(name: "__vla_expr0", {{.*}} flags: DIFlagArtificial
// CHECK-DAG: ![[VAR]] = !DILocalVariable(name: "vla",{{.*}} line: [[@LINE+2]]
// CHECK-DAG: !DISubrange(count: ![[VLAEXPR]])
  int vla[s];
  int i;
  for (i = 0; i < s; i++) {
    vla[i] = i*i;
  }
}
