// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OGCG

void use(int);

void host_teams(int x) {
#pragma omp teams
  {
    use(x);
  }
}

void teams_nested_parallel_for(void) {
#pragma omp teams
  {
#pragma omp parallel for
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// LLVM-LABEL: define {{.*}} void @host_teams(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @host_teams..omp_par, ptr {{.*}})

// LLVM-LABEL: define internal void @host_teams..omp_par(
// LLVM:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// The nested `parallel for` forks its own threads via __kmpc_fork_call from
// inside the teams-outlined function, so the wsloop's static-init/fini and
// barrier live in a second, further-outlined function.
// LLVM-LABEL: define {{.*}} void @teams_nested_parallel_for(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @teams_nested_parallel_for..omp_par.1)

// LLVM-LABEL: define internal void @teams_nested_parallel_for..omp_par.1(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @teams_nested_parallel_for..omp_par)

// LLVM-LABEL: define internal void @teams_nested_parallel_for..omp_par(
// LLVM:         call void @__kmpc_for_static_init_4u(
// LLVM:         call void @__kmpc_for_static_fini(
// LLVM:         call void @__kmpc_barrier(
// LLVM:         call void @use(i32

// OGCG-LABEL: define {{.*}} void @host_teams(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @host_teams.omp_outlined, ptr {{.*}})

// OGCG-LABEL: define internal void @host_teams.omp_outlined(
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         ret void

// OGCG-LABEL: define {{.*}} void @teams_nested_parallel_for(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @teams_nested_parallel_for.omp_outlined)

// OGCG-LABEL: define internal void @teams_nested_parallel_for.omp_outlined(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @teams_nested_parallel_for.omp_outlined.omp_outlined)

// OGCG-LABEL: define internal void @teams_nested_parallel_for.omp_outlined.omp_outlined(
// OGCG:         call void @__kmpc_for_static_init_4(
// OGCG:         call void @use(i32
// OGCG:         call void @__kmpc_for_static_fini(
