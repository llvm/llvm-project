// RUN: %clang_cc1 -emit-llvm -O0 -o - %s | FileCheck %s

kernel __attribute__((reqd_work_group_size(1,2,4))) void kernel1(int a) {}

kernel __attribute__((work_group_size_hint(8,16,32))) void kernel2(int a) {}

// CHECK: opencl.kernels = !{[[MDNODE0:![0-9]+]], [[MDNODE3:![0-9]+]]}

// CHECK: [[MDNODE0]] = metadata !{void (i32)* @kernel1, metadata [[MDNODE2:![0-9]+]]}
// CHECK: [[MDNODE2]] = metadata !{metadata !"reqd_work_group_size", i32 1, i32 2, i32 4}
// CHECK: [[MDNODE3]] = metadata !{void (i32)* @kernel2, metadata [[MDNODE5:![0-9]+]]}
// CHECK: [[MDNODE5]] = metadata !{metadata !"work_group_size_hint", i32 8, i32 16, i32 32}
