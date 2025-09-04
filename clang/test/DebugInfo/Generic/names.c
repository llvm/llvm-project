// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10.0 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck --check-prefix=APPLE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm -debug-info-kind=limited %s -o - -gpubnames | FileCheck --check-prefix=DEFAULT %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm -debug-info-kind=limited %s -o - -ggnu-pubnames | FileCheck --check-prefix=GNU %s

// CHECK: !DICompileUnit({{.*}}, nameTableKind: None
// DEFAULT-NOT: !DICompileUnit({{.*}}, nameTableKind:
// GNU: !DICompileUnit({{.*}}, nameTableKind: GNU
// APPLE: !DICompileUnit({{.*}}, nameTableKind: Apple

void f1(void) {
}
