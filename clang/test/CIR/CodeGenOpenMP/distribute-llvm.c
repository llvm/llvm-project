// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OGCG

void during(int);

// Explicit `teams` + `distribute` nesting lowers the same as the combined
// `teams distribute` spelling below.
void explicit_teams_distribute() {
#pragma omp teams
  {
#pragma omp distribute
    for (int i = 0; i < 10; i++) {
      during(i);
    }
  }
}

void teams_distribute() {
#pragma omp teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// LLVM-LABEL: define {{.*}} void @explicit_teams_distribute(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @explicit_teams_distribute..omp_par)

// LLVM-LABEL: define internal void @explicit_teams_distribute..omp_par(
// LLVM:         call void @__kmpc_for_static_init_4u(ptr {{.*}}, i32 {{.*}}, i32 92,
// LLVM:         call void @__kmpc_for_static_fini(
// LLVM:         call void @during(i32

// LLVM-LABEL: define {{.*}} void @teams_distribute(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @teams_distribute..omp_par)

// LLVM-LABEL: define internal void @teams_distribute..omp_par(
// LLVM:         call void @__kmpc_for_static_init_4u(ptr {{.*}}, i32 {{.*}}, i32 92,
// LLVM:         call void @__kmpc_for_static_fini(
// LLVM:         call void @during(i32

// OGCG-LABEL: define {{.*}} void @explicit_teams_distribute(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @explicit_teams_distribute.omp_outlined)

// OGCG-LABEL: define internal void @explicit_teams_distribute.omp_outlined(
// OGCG:         call void @__kmpc_for_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 92,
// OGCG:         call void @during(i32
// OGCG:         call void @__kmpc_for_static_fini(

// OGCG-LABEL: define {{.*}} void @teams_distribute(
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @teams_distribute.omp_outlined)

// OGCG-LABEL: define internal void @teams_distribute.omp_outlined(
// OGCG:         call void @__kmpc_for_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 92,
// OGCG:         call void @during(i32
// OGCG:         call void @__kmpc_for_static_fini(
