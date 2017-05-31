// REQUIRES: x86-registered-target
// RUN: %clang -target i386-apple-darwin10 -flto -S -g %s -o - | FileCheck %s
int global;
int main() { 
  static int localstatic;
  return 0;
}

// CHECK: !DIGlobalVariable(name: "localstatic"
// CHECK-NOT:               linkageName:
// CHECK-SAME:              line: 5,
// CHECK-SAME:              variable: i32* @main.localstatic
// CHECK: !DIGlobalVariable(name: "global"
// CHECK-NOT:               linkageName:
// CHECK-SAME:              line: 3,
// CHECK-SAME:              variable: i32* @global
