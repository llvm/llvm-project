// RUN: %clang_cc1 %s -triple spirv64-unknown-unknown -fclangir -emit-cir -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa -target-cpu gfx900 -fclangir -emit-llvm -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa -target-cpu gfx900 -emit-llvm -o - | FileCheck %s --check-prefix=LLVM

// In C, Sema accepts these attributes on a device kernel.
__attribute__((device_kernel, reqd_work_group_size(1, 2, 4),
               work_group_size_hint(8, 16, 32), vec_type_hint(int)))
void k(void) {}

// CIR-LABEL: cir.func {{.*}}@k()
// CIR-NOT: cir.cl.kernel_execution_metadata
// LLVM-LABEL: define{{.*}} amdgpu_kernel void @k()
// LLVM-NOT: !{{reqd_work_group_size|work_group_size_hint|vec_type_hint}}
