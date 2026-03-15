// RUN: %clang_cc1 -triple nvptx-unknown-unknown -o - -debug-info-kind=limited %s -emit-llvm | FileCheck %s

// CHECK: DICompileUnit({{.*}}, nameTableKind: None

void f1(void) {
}
