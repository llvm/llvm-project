// Verifies the omp-host-op-filtering pass runs in the CIR-to-LLVM device
// pipeline.//
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -fclangir -emit-llvm-bc %s -o %t-cir-host.bc
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-cir-host.bc \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM

void use(int);
int host_helper(int);

void f(int x) {
  // Host-only computation that must not leak into the device module.
  int y = host_helper(x);
  use(y);
#pragma omp target map(tofrom : x)
  {
    x = x + 1;
  }
}

// LLVM-LABEL: define hidden void @f(
// LLVM-NOT:     call i32 @host_helper
// LLVM-NOT:     call void @use
// LLVM:         ret void

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_f_l
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         %[[LD:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         %[[ADD:.*]] = add nsw i32 %[[LD]], 1
// LLVM:         store i32 %[[ADD]], ptr %{{.*}}, align 4
// LLVM:         call void @__kmpc_target_deinit()
// LLVM:         ret void
