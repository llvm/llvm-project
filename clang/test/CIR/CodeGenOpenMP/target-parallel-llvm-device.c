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

void use(int);

void target_parallel(int x) {
#pragma omp target parallel map(tofrom : x)
  {
    use(x);
  }
}

void target_parallel_proc_bind(int x) {
#pragma omp target parallel proc_bind(spread) map(tofrom : x)
  {
    use(x);
  }
}

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}..omp_par, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}..omp_par.wrapper, ptr {{.*}}, i64 1, i32 0)
// LLVM:         call void @__kmpc_target_deinit()
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}..omp_par(
// LLVM:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}..omp_par, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}..omp_par.wrapper, ptr {{.*}}, i64 1, i32 0)
// LLVM:         call void @__kmpc_target_deinit()
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}..omp_par(
// LLVM:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// OGCG-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_l
// OGCG:         call i32 @__kmpc_target_init(
// OGCG:       user_code.entry:
// OGCG:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}_omp_outlined, ptr null, ptr {{.*}}, i64 1, i32 0)
// OGCG:         call void @__kmpc_target_deinit()
// OGCG:         ret void

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}_omp_outlined(
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         ret void

// OGCG-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l
// OGCG:         call i32 @__kmpc_target_init(
// OGCG:       user_code.entry:
// OGCG:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}_omp_outlined, ptr null, ptr {{.*}}, i64 1, i32 0)
// OGCG:         call void @__kmpc_target_deinit()
// OGCG:         ret void

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}_omp_outlined(
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         ret void
