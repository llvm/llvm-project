// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -emit-llvm -w -o - %s | FileCheck %s

void t1 (void *f, int g) {
  // CHECK: call void asm "str $1, $0", "=*Q,r"
  asm("str %1, %0" : "=Q"(f) : "r"(g));
}
