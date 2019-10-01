// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -emit-llvm -o - -triple arm-none-linux-gnueabi -target-feature "+soft-float-abi" %s | FileCheck %s
// RUN: %clang_cc1 -verify -triple arm-none-linux-gnueabi -target-feature "+soft-float-abi" %s

// CHECK-NOT: +soft-float-abi

void foo() {}
__attribute__((target("crc"))) void bar() {}
__attribute__((target("soft-float-abi"))) void baz() {} // expected-warning{{unsupported 'soft-float-abi' in the 'target' attribute string}}
