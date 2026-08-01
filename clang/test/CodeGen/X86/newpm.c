// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple=x86_64-unkown-linux-gnu -fenable-new-pm-codegen -S -o - %s | FileCheck %s

int foo() {
  // CHECK-LABEL: foo
  // CHECK: xorl
  // CHECK: retq
  return 0;
}
