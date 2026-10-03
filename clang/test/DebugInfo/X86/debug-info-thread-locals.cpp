// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - -O0 | FileCheck %s

// Test that thread-local variables are emitted in correct scopes.

void test() {
  thread_local int bar = 2;
  {
    thread_local int bar = 1;
    {
      thread_local int bar = 0;
    }
  }
}

// CHECK: ![[FS_GVE:[0-9]+]] = !DIGlobalVariableExpression(var: ![[FS_GV:[0-9]+]]
// CHECK: ![[FS_GV]] = distinct !DIGlobalVariable(name: "bar", scope: ![[FSCOPE:[0-9]+]]
// CHECK: ![[FSCOPE]] = distinct !DISubprogram(name: "test"
// CHECK-SAME:                                retainedNodes: ![[FS_DECLS:[0-9]+]]
// CHECK: ![[FS_DECLS]] = !{![[FS_GVE]], ![[LB1_GVE:[0-9]+]], ![[LB2_GVE:[0-9]+]]}
// CHECK: ![[LB1_GVE]] = !DIGlobalVariableExpression(var: ![[LB1_GV:[0-9]+]]
// CHECK: ![[LB1_GV]] = distinct !DIGlobalVariable(name: "bar", scope: ![[LB1SCOPE:[0-9]+]]
// CHECK: ![[LB1SCOPE]] = distinct !DILexicalBlock(scope: ![[FSCOPE]]
// CHECK: ![[LB2_GVE]] = !DIGlobalVariableExpression(var: ![[LB2_GV:[0-9]+]]
// CHECK: ![[LB2_GV]] = distinct !DIGlobalVariable(name: "bar", scope: ![[LB2SCOPE:[0-9]+]]
// CHECK: ![[LB2SCOPE]] = distinct !DILexicalBlock(scope: ![[LB1SCOPE]]
