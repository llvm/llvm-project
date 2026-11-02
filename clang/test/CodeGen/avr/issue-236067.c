// RUN: %clang_cc1 -triple avr -emit-llvm < %s | FileCheck %s

// This test case verifies https://github.com/llvm/llvm-project/issues/236067 .

void foo() {
  void *bar = &&label;
label:
}

// CHECK: store ptr addrspacecast (ptr addrspace(1) blockaddress(@foo, %label) to ptr), ptr %bar, align 1
