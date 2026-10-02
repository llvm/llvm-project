// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
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

// The explicit target/parallel/for nesting and the combined `target parallel
// for` spelling lower identically: an omp.target forks an omp.parallel via
// __kmpc_fork_call, whose body statically schedules the omp.wsloop via
// __kmpc_for_static_init_4/_4u.
//
// LLVM-LABEL: define {{.*}} void @target_parallel_for(
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}(

// LLVM-LABEL: define {{.*}} void @combined_target_parallel_for(
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}(

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par)

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par(
// LLVM:         call void @__kmpc_for_static_init_4u(
// LLVM:         call void @__kmpc_for_static_fini(
// LLVM:         call void @__kmpc_barrier(
// LLVM:         call void @during(i32

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par)

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par(
// LLVM:         call void @__kmpc_for_static_init_4u(
// LLVM:         call void @__kmpc_for_static_fini(
// LLVM:         call void @__kmpc_barrier(
// LLVM:         call void @during(i32

// OGCG-LABEL: define {{.*}} void @target_parallel_for(
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}(

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}.omp_outlined)

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}.omp_outlined(
// OGCG:         call void @__kmpc_for_static_init_4(
// OGCG:         call void @during(i32
// OGCG:         call void @__kmpc_for_static_fini(

// OGCG-LABEL: define {{.*}} void @combined_target_parallel_for(
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}(

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}.omp_outlined)

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}.omp_outlined(
// OGCG:         call void @__kmpc_for_static_init_4(
// OGCG:         call void @during(i32
// OGCG:         call void @__kmpc_for_static_fini(
