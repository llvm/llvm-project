// RUN: %clang_cc1 -triple i686-windows-win32 -fms-extensions -debug-info-kind=limited -emit-llvm %s -o - \
// RUN:    | FileCheck %s

struct __declspec(dllexport) s {
  static const unsigned int ui = 0;
};

// CHECK: ![[SCOPE:[0-9]+]] = distinct !DICompileUnit(
// CHECK: !DIGlobalVariable(name: "ui", linkageName: "\01?ui@s@@2IB", scope: ![[SCOPE]],
// CHECK-SAME:              variable: i32* @"\01?ui@s@@2IB"

