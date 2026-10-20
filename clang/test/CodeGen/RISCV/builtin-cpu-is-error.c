// RUN: not %clang_cc1 -triple riscv64-unknown-linux-gnu -emit-llvm %s -o - 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error: invalid cpu name for builtin
int test_cpu_is_invalid_cpu() {
  return __builtin_cpu_is("generic-rv64");
}
