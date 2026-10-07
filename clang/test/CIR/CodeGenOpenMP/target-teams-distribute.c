// REQUIRES: amdgpu-registered-target

// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas live in the private address space.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

void during(int);

// `target teams distribute` stays a generic (non-SPMD) kernel_type, but
// still needs its trip count host_eval'd.
void explicit_target_teams_distribute() {
#pragma omp target
  {
#pragma omp teams
    {
#pragma omp distribute
      for (int i = 0; i < 10; i++) {
        during(i);
      }
    }
  }
}

// CIR-HOST: cir.func{{.*}}@explicit_target_teams_distribute
// CIR-HOST: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
// CIR-HOST: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32
// CIR-HOST: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
// CIR-HOST: omp.target kernel_type(generic) host_eval(%[[ZERO]] -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %[[STEP]] -> %[[ASTEP:.*]] : i32, i32, i32) {
// CIR-HOST:   omp.teams {
// CIR-HOST:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST:     omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:  omp.loop_nest (%{{.*}}) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[ASTEP]]) {
// CIR-HOST:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.call @{{.*}}during
// CIR-HOST:          omp.yield
// CIR-HOST:        }
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@explicit_target_teams_distribute
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE:   omp.teams {
// CIR-DEVICE:     omp.distribute private(@{{.*}} %{{.*}} -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:  omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CIR-DEVICE:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.call @{{.*}}during
// CIR-DEVICE:          omp.yield
// CIR-DEVICE:        }
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

// The combined `target teams distribute` spelling decomposes into the same
// target/teams/distribute directive nesting as above, so it lowers
// identically.
void target_teams_distribute() {
#pragma omp target teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_distribute
// CIR-HOST: omp.target kernel_type(generic) host_eval(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : i32, i32, i32) {
// CIR-HOST:   omp.teams {
// CIR-HOST:     omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:  omp.loop_nest
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_distribute
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE:   omp.teams {
// CIR-DEVICE:     omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:  omp.loop_nest
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}
