// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - -O0 | FileCheck %s

// Test that static local variables are emitted in correct scopes.

void test() {
  static int bar = 2;
  {
    static int bar = 1;
    {
      static int bar = 0;
    }
  }
}

// Automatic local variables in braceless arms of
// an if statement are placed in separate DILexicalBlocks.
// Test that the behavior for static locals is the same as
// for automatic variables in that case.
void test_braceless(int x) {
  if (x)
    static int foo = 30;
  else
    static int foo = 40;
  if (x)
#line 100
    int foobar = 50;
  else
#line 200
    int foobar = 60;
  if (true)
    int folded = x;
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

// CHECK: ![[B_LB1_GVE:[0-9]+]] = !DIGlobalVariableExpression(var: ![[B_LB1_GV:[0-9]+]]
// CHECK: ![[B_LB1_GV]] = distinct !DIGlobalVariable(name: "foo", scope: ![[THEN_SCOPE:[0-9]+]],
// CHECK: ![[THEN_SCOPE]] = distinct !DILexicalBlock(scope: ![[STATIC_IF:[0-9]+]],
// CHECK: ![[STATIC_IF]] = distinct !DILexicalBlock(scope: ![[BSCOPE:[0-9]+]],
// CHECK: ![[BSCOPE]] = distinct !DISubprogram(name: "test_braceless"
// CHECK-SAME:                                retainedNodes: ![[B_DECLS:[0-9]+]]
// CHECK: ![[B_DECLS]] = !{![[B_LB1_GVE]], ![[B_LB2_GVE:[0-9]+]]}
// CHECK: ![[B_LB2_GVE]] = !DIGlobalVariableExpression(var: ![[B_LB2_GV:[0-9]+]]
// CHECK: ![[B_LB2_GV]] = distinct !DIGlobalVariable(name: "foo", scope: ![[ELSE_SCOPE:[0-9]+]],
// CHECK: ![[ELSE_SCOPE]] = distinct !DILexicalBlock(scope: ![[STATIC_IF]],

// CHECK: ![[AUTO_IF:[0-9]+]] = distinct !DILexicalBlock(scope: ![[BSCOPE]],
// CHECK: !DILocalVariable(name: "foobar", scope: ![[AUTO_THEN:[0-9]+]],
// CHECK-SAME:                                      line: 100
// CHECK: ![[AUTO_THEN]] = distinct !DILexicalBlock(scope: ![[AUTO_IF]],
// CHECK: !DILocalVariable(name: "foobar", scope: ![[AUTO_ELSE:[0-9]+]],
// CHECK-SAME:                                      line: 200
// CHECK: ![[AUTO_ELSE]] = distinct !DILexicalBlock(scope: ![[AUTO_IF]],

// CHECK: !DILocalVariable(name: "folded", scope: ![[FOLDED_ARM:[0-9]+]],
// CHECK: ![[FOLDED_ARM]] = distinct !DILexicalBlock(scope: ![[FOLDED_IF:[0-9]+]],
// CHECK: ![[FOLDED_IF]] = distinct !DILexicalBlock(scope: ![[BSCOPE]],
