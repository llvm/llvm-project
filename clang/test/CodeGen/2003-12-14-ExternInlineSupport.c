// RUN: %clang_cc1 -triple %itanium_abi_triple -std=gnu89 %s -emit-llvm -o - | FileCheck %s

extern __inline__ void dead_function(void) {}
// CHECK-NOT: dead_function
