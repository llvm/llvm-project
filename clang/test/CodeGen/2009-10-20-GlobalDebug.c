// REQUIRES: x86-registered-target
// RUN: %clang -target i386-apple-darwin10 -flto -S -g %s -o - | FileCheck %s

// CHECK: @main.localstatic = internal global i32 0, align 4, !dbg [[L:![0-9]+]]
// CHECK: @global ={{.*}} global i32 0, align 4, !dbg [[G:![0-9]+]]

int global;
int main() {
  static int localstatic;
  return 0;
}

// CHECK: [[L]] = !DIGlobalVariableExpression(var: [[LV:.*]], expr: !DIExpression())
// CHECK: [[LV]] = distinct !DIGlobalVariable(name: "localstatic"
// CHECK-NOT:                                 linkageName:
// CHECK-SAME:                                line: 9,
// CHECK: [[G]] = !DIGlobalVariableExpression(var: [[GV:.*]], expr: !DIExpression())
// CHECK: [[GV]] = distinct !DIGlobalVariable(name: "global"
// CHECK-NOT:                                 linkageName:
// CHECK-SAME:                                line: 7,
