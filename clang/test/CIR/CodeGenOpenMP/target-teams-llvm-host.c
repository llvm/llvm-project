// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OGCG

void use(int);

void explicit_target_teams(int x) {
#pragma omp target map(tofrom : x)
#pragma omp teams
  {
    use(x);
  }
}

void target_teams(int x) {
#pragma omp target teams map(tofrom : x)
  {
    use(x);
  }
}

// Host wrappers
//
// LLVM-LABEL: define {{.*}} void @explicit_target_teams(
// LLVM-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM:         store i32 %[[ARG]], ptr %[[X_ADDR]], align 4
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_explicit_target_teams_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// LLVM-LABEL: define {{.*}} void @target_teams(
// LLVM-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM:         store i32 %[[ARG]], ptr %[[X_ADDR]], align 4
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_teams_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// Outlined target functions: both the explicit `target`+`teams` nesting and
// the combined `target teams` spelling lower identically, forking an
// omp.teams region via __kmpc_fork_teams.
//
// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_explicit_target_teams_l{{.*}}..omp_par, ptr {{.*}})

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_l{{.*}}..omp_par(
// LLVM:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_target_teams_l{{.*}}..omp_par, ptr {{.*}})

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_l{{.*}}..omp_par(
// LLVM:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// OGCG interleaves host wrapper and outlined function per target region.
//
// OGCG-LABEL: define {{.*}} void @explicit_target_teams(i32
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_explicit_target_teams_l

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_l{{.*}}(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_explicit_target_teams_l{{.*}}.omp_outlined, ptr {{.*}})

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_l{{.*}}.omp_outlined(
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         ret void

// OGCG-LABEL: define {{.*}} void @target_teams(i32
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_target_teams_l

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_l{{.*}}(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_target_teams_l{{.*}}.omp_outlined, ptr {{.*}})

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_l{{.*}}.omp_outlined(
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         ret void
