// REQUIRES: riscv-registered-target
// RUN: not %clang_cc1 -triple riscv64 -target-feature +zifencei -target-feature +m -target-feature +a \
// RUN:  -emit-llvm %s 2>&1 | FileCheck %s

// CHECK: error: duplicate 'arch=' in the 'target' attribute string;
__attribute__((target("arch=rv64gc;arch=rv64gc_zbb"))) void testMultiArchSelectLast() {}
// CHECK: error: duplicate 'cpu=' in the 'target' attribute string;
__attribute__((target("cpu=sifive-u74;cpu=sifive-u54"))) void testMultiCpuSelectLast() {}
// CHECK: error: duplicate 'tune=' in the 'target' attribute string;
__attribute__((target("tune=sifive-u74;tune=sifive-u54"))) void testMultiTuneSelectLast() {}
