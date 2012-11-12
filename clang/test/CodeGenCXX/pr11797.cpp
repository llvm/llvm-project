// RUN: %clang_cc1 %s -fvisibility hidden -emit-llvm -o - | FileCheck %s

namespace std __attribute__ ((__visibility__ ("default"))) {}
#pragma GCC visibility push(default)
void foo() {
}
#pragma GCC visibility pop
// CHECK: define void @_Z3foov()
