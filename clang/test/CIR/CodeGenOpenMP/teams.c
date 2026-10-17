// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void use(int);

// A standalone `teams` directive lowers to an omp.teams op whose region holds
// the associated structured block.
void host_teams(int x) {
  // CHECK: cir.func{{.*}}@host_teams
  // CHECK: omp.teams {
  // CHECK: cir.call @use
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp teams
  {
    use(x);
  }
}

void teams_nested_parallel_for(void) {
  // CHECK: cir.func{{.*}}@teams_nested_parallel_for
  // CHECK: omp.teams {
  // CHECK: omp.parallel {
  // CHECK: omp.wsloop private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest
  // CHECK: cir.call @use
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: } {omp.combined}
  // CHECK: omp.terminator
  // CHECK: } {omp.combined}
#pragma omp teams
  {
#pragma omp parallel for
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}
