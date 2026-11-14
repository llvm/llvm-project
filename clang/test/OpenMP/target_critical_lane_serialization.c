// On a GPU, whose threads of a warp may run in lockstep, a critical region is
// run by the threads of each warp one at a time, so that only one of them
// contends for the critical lock.

// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host-nv.bc
// RUN: %clang_cc1 -fopenmp -x c -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-nv.bc -o - | FileCheck %s
// expected-no-diagnostics

#pragma omp declare target
int X;

// CHECK-LABEL: define {{.*}}void @critical_region(
// CHECK:       %omp.lanes.mask = call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       omp.lanes.header:
// CHECK:       omp.lanes.body:
// CHECK:         call void @__kmpc_critical(
// CHECK:         call void @__kmpc_end_critical(
// CHECK:       omp.lanes.latch:
// CHECK-NEXT:    call void @__kmpc_syncwarp(i64 %omp.lanes.mask)
// CHECK:       omp.lanes.exit:
// CHECK-NOT:   omp.critical.loop
// CHECK:       ret void
void critical_region(void) {
#pragma omp critical
  X++;
}
#pragma omp end declare target
