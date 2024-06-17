// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s

// CHECK: name: "__amdgpu_buffer_rsrc_t",{{.*}}baseType: ![[BT:[0-9]+]]
// CHECK: [[BT]] = !DICompositeType(tag: DW_TAG_structure_type, name: "__amdgpu_buffer_rsrc_t", {{.*}} flags: DIFlagFwdDecl)
void test_locals(void) {
  __amdgpu_buffer_rsrc_t k;
}
