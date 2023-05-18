__attribute__((overloadable))
void testRiscv(__rvv_int8mf8_t);
// CHECK: USR: c:@F@testRiscv#@BT@__rvv_int8mf8_t#

// RUN: c-index-test -index-file %s --target=riscv64 -target-feature +v | FileCheck %s
