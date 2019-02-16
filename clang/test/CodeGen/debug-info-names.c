// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - -gpubnames | FileCheck --check-prefix=DEFAULT %s
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - -ggnu-pubnames | FileCheck --check-prefix=GNU %s

// CHECK: !DICompileUnit({{.*}}, nameTableKind: None
// DEFAULT-NOT: !DICompileUnit({{.*}}, nameTableKind:
// GNU: !DICompileUnit({{.*}}, nameTableKind: GNU

void f1() {
}
