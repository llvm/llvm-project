// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s
// llvm.sqrt has undefined behavior on negative inputs, so it is
// inappropriate to translate C/C++ sqrt to this.
float sqrtf(float x);
float foo(float X) {
  // CHECK: foo
  // CHECK: call float @sqrtf(float %
  // Check that this is marked readonly when errno is ignored.
  return sqrtf(X);
}
