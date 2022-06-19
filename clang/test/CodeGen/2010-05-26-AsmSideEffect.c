// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -triple arm-apple-darwin -o - | FileCheck %s
// Radar 8026855

int test (void *src) {
  register int w0 asm ("0");
  // CHECK: call i32 asm "ldr $0, [$1]", "={r0}{{.*}}(i8*
  asm ("ldr %0, [%1]": "=r" (w0): "r" (src));
  return w0;
}
