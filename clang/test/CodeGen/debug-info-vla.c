// REQUIRES: rdar42833777
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

void testVLAwithSize(int s)
{
<<<<<<< HEAD
// CHECK-DAG: dbg.declare({{.*}} %__vla_expr, metadata ![[VLAEXPR:[0-9]+]]
// CHECK-DAG: dbg.declare({{.*}} %vla, metadata ![[VAR:[0-9]+]]
// CHECK-DAG: ![[VLAEXPR]] = !DILocalVariable(name: "__vla_expr", {{.*}} flags: DIFlagArtificial
// CHECK-DAG: ![[VAR]] = !DILocalVariable(name: "vla",{{.*}} line: [[@LINE+2]]
// CHECK-DAG: !DISubrange(count: ![[VLAEXPR]])
||||||| 87815378b0e... Recommit rL323952: [DebugInfo] Enable debug information for C99 VLA types.
// CHECK-DAG: dbg.declare({{.*}} %vla_expr, metadata ![[VLAEXPR:[0-9]+]]
// CHECK-DAG: dbg.declare({{.*}} %vla, metadata ![[VAR:[0-9]+]]
// CHECK-DAG: ![[VLAEXPR]] = !DILocalVariable(name: "vla_expr"
// CHECK-DAG: ![[VAR]] = !DILocalVariable(name: "vla",{{.*}} line: [[@LINE+2]]
// CHECK-DAG: !DISubrange(count: ![[VLAEXPR]])
=======
// CHECK: dbg.declare
// CHECK: dbg.declare({{.*}}, metadata ![[VAR:.*]], metadata !DIExpression())
// CHECK: ![[VAR]] = !DILocalVariable(name: "vla",{{.*}} line: [[@LINE+1]]
>>>>>>> parent of 87815378b0e... Recommit rL323952: [DebugInfo] Enable debug information for C99 VLA types.
  int vla[s];
  int i;
  for (i = 0; i < s; i++) {
    vla[i] = i*i;
  }
}
