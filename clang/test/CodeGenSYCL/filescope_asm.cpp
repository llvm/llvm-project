// RUN:  %clang_cc1 -fsycl-is-device -triple spir64 -emit-llvm %s -o - | FileCheck %s
//
// Check that file-scope asm is ignored during device-side SYCL compilation.
//
// CHECK-NOT: module asm "foo"
__asm__("foo");
