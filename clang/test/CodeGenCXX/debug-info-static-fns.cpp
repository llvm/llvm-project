// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

namespace A {
  static int a(int b) { return b + 4; }

  int b(int c) { return c + a(c); }
}

// Verify that a is present and mangled.
// CHECK: define internal i32 @_ZN1AL1aEi({{.*}} !dbg [[DBG:![0-9]+]]
// CHECK: [[DBG]] = distinct !DISubprogram(name: "a", linkageName: "_ZN1AL1aEi",
// CHECK-SAME:          line: 4
// CHECK-SAME:          DISPFlagDefinition
