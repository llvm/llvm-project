// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

void testVLAwithSize(int s)
{
// CHECK: dbg.declare
// CHECK: dbg.declare({{.*}}, metadata ![[VAR:.*]], metadata !DIExpression())
// CHECK: ![[VAR]] = !DILocalVariable(name: "vla",{{.*}} line: [[@LINE+1]]
  int vla[s];
  int i;
  for (i = 0; i < s; i++) {
    vla[i] = i*i;
  }
}
