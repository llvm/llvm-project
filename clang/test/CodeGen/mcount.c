// RUN: %clang_cc1 -pg -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
void foo(void) {
// CHECK: call void @mcount()
}
