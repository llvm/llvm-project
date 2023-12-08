// RUN: echo "#include <stddef.h>" > %t.h
// RUN: %clang_cc1 -S -debug-info-kind=limited -include %t.h %s -emit-llvm -o - | FileCheck %s

// CHECK: !DIGlobalVariable(name: "outer",
// CHECK-NOT:               linkageName:
// CHECK-SAME:              line: [[@LINE+2]]
// CHECK-SAME:              isDefinition: true
int outer = 42;

