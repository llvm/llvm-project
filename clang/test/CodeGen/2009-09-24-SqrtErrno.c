// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -fmath-errno | FileCheck %s
// llvm.sqrt has undefined behavior on negative inputs, so it is
// inappropriate to translate C/C++ sqrt to this.

float sqrtf(float x);
float foo(float X) {
// CHECK: foo
// CHECK-NOT: readonly
// CHECK: call float @sqrtf
  // Check that this is not marked readonly when errno is used.
  return sqrtf(X);
}
