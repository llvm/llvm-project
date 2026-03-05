// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o - 2>&1 | FileCheck %s
// XFAIL: *

void f() {}

void reject() {
  for (int i = 0; i < 100; i++, f());
  // CHECK: failed to legalize operation 'cir.scope'
}
