// RUN: %clang_cc1 -O1 -triple spirv64 -fsycl-is-device -x c++ %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -O1 -triple spirv64 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -O1 -triple spirv32 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: define spir_func void @{{.*}}test_group_barrier{{.*}}(
// CHECK-SAME: ) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    tail call void @llvm.spv.group.memory.barrier.with.group.sync()
// CHECK-NEXT:    ret void
//
[[clang::sycl_external]] void test_group_barrier() {
  __builtin_spirv_group_barrier();
}
