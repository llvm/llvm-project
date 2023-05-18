// REQUIRES: amdgpu-registered-target
//
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -O0 -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,ON %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -O0 -emit-llvm -o - %s \
// RUN:   -mno-amdgpu-ieee -menable-no-nans \
// RUN:   | FileCheck -check-prefixes=COMMON,OFF %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -O0 -emit-llvm -o - %s \
// RUN:   -mno-amdgpu-ieee -cl-fast-relaxed-math \
// RUN:   | FileCheck -check-prefixes=COMMON,OFF %s

// Check AMDGCN ISA generation.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -O3 -S -o - %s \
// RUN:   | FileCheck -check-prefixes=ISA-ON %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -O3 -S -o - %s \
// RUN:   -mno-amdgpu-ieee -menable-no-nans \
// RUN:   | FileCheck -check-prefixes=ISA-OFF %s

// Check diagnostics when using -mno-amdgpu-ieee without NoHonorNaNs.

// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -O0 -emit-llvm -o - %s \
// RUN:   -mno-amdgpu-ieee 2>&1 | FileCheck -check-prefixes=DIAG %s

// COMMON: define{{.*}} amdgpu_kernel void @kern{{.*}} [[ATTRS1:#[0-9]+]]
// ISA-ON: v_mul_f32_e64 v{{[0-9]+}}, 1.0, s{{[0-9]+}}
// ISA-ON: v_mul_f32_e64 v{{[0-9]+}}, 1.0, s{{[0-9]+}}
// ISA-ON: v_min_f32_e32
// ISA-ON: ; IeeeMode: 1
// ISA-OFF-NOT: v_mul_f32_e64 v{{[0-9]+}}, 1.0, s{{[0-9]+}}
// ISA-OFF-NOT: v_mul_f32_e64 v{{[0-9]+}}, 1.0, s{{[0-9]+}}
// ISA-OFF: v_min_f32_e32
// ISA-OFF: ; IeeeMode: 0
kernel void kern(global float *x, float y, float z) {
  *x = __builtin_fmin(y, z);
}

// COMMON: define{{.*}}void @fun() [[ATTRS2:#[0-9]+]]
void fun() {
}

// ON-NOT: attributes [[ATTRS1]] = {{.*}} "amdgpu-ieee"
// OFF: attributes [[ATTRS1]] = {{.*}} "amdgpu-ieee"="false"{{.*}}"no-nans-fp-math"="true"{{.*}}"no-trapping-math"="true"
// ON-NOT: attributes [[ATTRS2]] = {{.*}} "amdgpu-ieee"
// OFF: attributes [[ATTRS2]] = {{.*}} "amdgpu-ieee"="false"{{.*}}"no-nans-fp-math"="true"{{.*}}"no-trapping-math"="true"

// DIAG: invalid argument '-mno-amdgpu-ieee' only allowed with relaxed NaN handling
