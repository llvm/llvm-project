// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
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

// Host wrappers
//
// LLVM-LABEL: define {{.*}} void @target_parallel(
// LLVM-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM:         store i32 %[[ARG]], ptr %[[X_ADDR]], align 4
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// LLVM-LABEL: define {{.*}} void @target_parallel_proc_bind(
// LLVM-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM:         store i32 %[[ARG]], ptr %[[X_ADDR]], align 4
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// Outlined target functions: `target parallel` forks an omp.parallel region
// via __kmpc_fork_call; the proc_bind clause lowers to a __kmpc_push_proc_bind
// call right before the fork.
//
// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}..omp_par, ptr {{.*}})

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}..omp_par(
// LLVM:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         call void @__kmpc_push_proc_bind(ptr {{.*}}, i32 {{.*}}, i32 4)
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}..omp_par, ptr {{.*}})

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}..omp_par(
// LLVM:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// OGCG interleaves host wrapper and outlined function per target region.
//
// OGCG-LABEL: define {{.*}} void @target_parallel(i32
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_target_parallel_l

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}.omp_outlined, ptr {{.*}})

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}.omp_outlined(
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         ret void

// OGCG-LABEL: define {{.*}} void @target_parallel_proc_bind(i32
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}(
// OGCG:         call void @__kmpc_push_proc_bind(ptr {{.*}}, i32 {{.*}}, i32 4)
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}.omp_outlined, ptr {{.*}})

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}.omp_outlined(
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         ret void
