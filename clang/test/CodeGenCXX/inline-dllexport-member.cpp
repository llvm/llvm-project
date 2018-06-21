// RUN: %clang_cc1 -triple i686-windows-win32 -fms-extensions -debug-info-kind=limited -emit-llvm %s -o - \
// RUN:    | FileCheck %s

// CHECK: @"\01?ui@s@@2IB" = weak_odr dllexport constant i32 0, comdat, align 4, !dbg [[UI:![0-9]+]]

struct __declspec(dllexport) s {
  static const unsigned int ui = 0;
};

// CHECK: [[UI]] = !DIGlobalVariableExpression(var: [[UIV:.*]], expr: !DIExpression())
// CHECK: [[UIV]] = distinct !DIGlobalVariable(name: "ui", linkageName: "\01?ui@s@@2IB", scope: ![[SCOPE:[0-9]+]],
// CHECK: ![[SCOPE]] = distinct !DICompileUnit(

