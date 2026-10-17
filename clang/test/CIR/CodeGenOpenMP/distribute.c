// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void during(int);

// Explicit `teams` + `distribute` nesting lowers the same as the combined
// `teams distribute` spelling below, so both get omp.combined.
void explicit_teams_distribute() {
  // CHECK: cir.func{{.*}}@explicit_teams_distribute
#pragma omp teams
  {
#pragma omp distribute
    for (int i = 0; i < 10; i++) {
      during(i);
    }
  }

  // CHECK: omp.teams {
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
  // CHECK: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CHECK: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CHECK: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32

  // CHECK: omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.load align(4) %[[I_PRIV]]
  // CHECK: cir.call @{{.*}}during
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: } {omp.combined}
}

// Combined `teams distribute` lowers the same as the explicit nesting
// above, with omp.combined on the omp.teams.
void teams_distribute() {
  // CHECK: cir.func{{.*}}@teams_distribute
#pragma omp teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }

  // CHECK: omp.teams {
  // CHECK: omp.distribute private(@{{.*}} %{{.*}} -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: } {omp.combined}
}
