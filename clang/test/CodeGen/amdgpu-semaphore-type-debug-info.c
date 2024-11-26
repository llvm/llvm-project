// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s

// CHECK: name: "__amdgpu_semaphore0_t",{{.*}}baseType: ![[BT:[0-9]+]]
// CHECK: [[BT]] = !DIBasicType(name: "__amdgpu_semaphore0_t", size: 128, encoding: DW_ATE_unsigned)
// CHECK: name: "__amdgpu_semaphore7_t",{{.*}}baseType: ![[BT:[0-9]+]]
// CHECK: [[BT]] = !DIBasicType(name: "__amdgpu_semaphore7_t", size: 128, encoding: DW_ATE_unsigned)
void test_locals(void) {
  __amdgpu_semaphore0_t *k0;
  __amdgpu_semaphore7_t *k7;
}
