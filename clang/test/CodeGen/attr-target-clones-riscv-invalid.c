// RUN: not %clang_cc1 -triple riscv64 -target-feature +i -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK-UNSUPPORT-OS

// CHECK-UNSUPPORT-OS: error: function multiversioning is currently only supported on Linux
__attribute__((target_clones("default", "arch=+c"))) int foo(void) {
  return 2;
}

int bar() { return foo(); }
