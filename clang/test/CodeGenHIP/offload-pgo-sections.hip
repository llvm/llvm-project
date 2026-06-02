// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// Verify CGCUDANV emits the per-TU __llvm_profile_sections_<CUID> global
// for HIP+PGO compilations. Device subcompile: populated 7-pointer struct
// in addrspace(1). Host compile: void* shadow registered with the HIP
// runtime and with the profile runtime's drain list.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefix=DEV %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefix=HOST %s

// Guard: no PGO -> no emission.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefix=NONE %s

// Guard: no CUID -> no emission.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefix=NONE %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ int helper(int x) { return x + 1; }
__global__ void kernel(int *p) { *p = helper(*p); }

// DEV-DAG: @__start___llvm_prf_names = external hidden addrspace(1) global i8
// DEV-DAG: @__stop___llvm_prf_names = external hidden addrspace(1) global i8
// DEV-DAG: @__start___llvm_prf_cnts = external hidden addrspace(1) global i8
// DEV-DAG: @__stop___llvm_prf_cnts = external hidden addrspace(1) global i8
// DEV-DAG: @__start___llvm_prf_data = external hidden addrspace(1) global i8
// DEV-DAG: @__stop___llvm_prf_data = external hidden addrspace(1) global i8
// DEV-DAG: @__llvm_profile_raw_version = external addrspace(1) constant i64
// DEV: @__llvm_profile_sections_[[CUID:[0-9a-f]+]] = protected addrspace(1) constant {{.*}}@__start___llvm_prf_names{{.*}}@__stop___llvm_prf_names{{.*}}@__start___llvm_prf_cnts{{.*}}@__stop___llvm_prf_cnts{{.*}}@__start___llvm_prf_data{{.*}}@__stop___llvm_prf_data{{.*}}@__llvm_profile_raw_version
// DEV: @llvm.compiler.used = {{.*}}@__llvm_profile_sections_[[CUID]]

// HOST: @__llvm_profile_sections_[[CUID:[0-9a-f]+]] = global ptr null
// HOST: @llvm.compiler.used = {{.*}}@__llvm_profile_sections_[[CUID]]
// HOST: define internal void @__hip_register_globals
// HOST: call void @__hipRegisterVar({{.*}}@__llvm_profile_sections_[[CUID]],
// HOST: call void @__llvm_profile_offload_register_shadow_variable(ptr @__llvm_profile_sections_[[CUID]])

// NONE-NOT: __llvm_profile_sections_
// NONE-NOT: __llvm_profile_offload_register_shadow_variable
