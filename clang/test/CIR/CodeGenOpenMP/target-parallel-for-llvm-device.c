// Two-step host-BC -> device pipeline that mirrors the offloading driver.
//
// Step 1: Host compilation to bitcode (provides offload entry info to device pass).
// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -fclangir -emit-llvm-bc %s -o %t-cir-host.bc
// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -emit-llvm-bc %s -o %t-ogcg-host.bc
//
// Step 2: Device compilation using host BC.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-cir-host.bc \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-ogcg-host.bc \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OGCG

void during(int);

void target_parallel_for() {
#pragma omp target
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

void combined_target_parallel_for() {
#pragma omp target parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_for_l
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par{{.*}}, ptr null, ptr {{.*}}, i64 0, i32 0)
// LLVM:         call void @__kmpc_target_deinit()

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par
// LLVM:         call void @__kmpc_for_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par,
// LLVM:         call void @__kmpc_barrier(

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par(i32
// LLVM:         call void @during(i32

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_combined_target_parallel_for_l
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par{{.*}}, ptr null, ptr {{.*}}, i64 0, i32 0)
// LLVM:         call void @__kmpc_target_deinit()

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par
// LLVM:         call void @__kmpc_for_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par,
// LLVM:         call void @__kmpc_barrier(

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par(i32
// LLVM:         call void @during(i32

// OGCG-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_for_l
// OGCG:         call i32 @__kmpc_target_init(
// OGCG:       user_code.entry:
// OGCG:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}_omp_outlined, ptr null, ptr {{.*}}, i64 0, i32 0)
// OGCG:         call void @__kmpc_target_deinit()

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}_omp_outlined(
// OGCG:         call void @__kmpc_for_static_init_4(
// OGCG:         call void @during(i32
// OGCG:         call void @__kmpc_for_static_fini(
// OGCG:         call void @__kmpc_barrier(

// OGCG-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_combined_target_parallel_for_l
// OGCG:         call i32 @__kmpc_target_init(
// OGCG:       user_code.entry:
// OGCG:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}_omp_outlined, ptr null, ptr {{.*}}, i64 0, i32 0)
// OGCG:         call void @__kmpc_target_deinit()

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}_omp_outlined(
// OGCG:         call void @__kmpc_for_static_init_4(
// OGCG:         call void @during(i32
// OGCG:         call void @__kmpc_for_static_fini(
