// RUN:  %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s
//
// Check that file-scope asm is ignored during device-side SYCL compilation.
//
// CHECK-NOT: module asm "foo"
__asm__("foo");
