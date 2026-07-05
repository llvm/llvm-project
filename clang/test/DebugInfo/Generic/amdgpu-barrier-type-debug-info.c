// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s

// CHECK: name: "__amdgpu_named_workgroup_barrier_t",{{.*}}baseType: ![[BT:[0-9]+]]
// CHECK: [[BT]] = !DIBasicType(name: "__amdgpu_named_workgroup_barrier_t", size: 128, encoding: DW_ATE_unsigned)
void test_locals(void) {
  __amdgpu_named_workgroup_barrier_t *k0;
}
