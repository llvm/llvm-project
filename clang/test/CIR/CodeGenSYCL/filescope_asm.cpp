// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Check that file-scope asm is ignored during device-side SYCL compilation.

// CIR: cir.module_asm = []
// LLVM-NOT: module asm "foo"
__asm__("foo");
