// REQUIRES: amdgpu-registered-target

// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas live in the private address space.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

// Host-side lowering to LLVM IR: identical on both paths, so LLVM-HOST
// checks both. The only difference is where the second function's helper
// functions end up relative to the first function's, checked below under
// LLVM-HOST-CIR/LLVM-HOST-OGCG.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-HOST,LLVM-HOST-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-HOST,LLVM-HOST-OGCG

// Device-side lowering to LLVM IR: compiled in two steps to mirror the
// offloading driver.
// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -fclangir -emit-llvm-bc %s -o %t-cir-host.bc
// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -emit-llvm-bc %s -o %t-ogcg-host.bc
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-cir-host.bc \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-DEVICE,LLVM-DEVICE-CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-ogcg-host.bc \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-DEVICE,LLVM-DEVICE-OGCG

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

// CIR-HOST: cir.func{{.*}}@explicit_target_teams_distribute(
// The loop only runs if its precondition `0 < 10` holds.
// CIR-HOST-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The host computes the trip count before the omp.target and passes it
// in through host_eval.
// tripCount = (10 - (0 - 1 + 1)) / 1 = 10
// CIR-HOST-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-HOST-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-HOST-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-HOST-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-HOST-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-HOST-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-HOST-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: omp.target kernel_type(generic) host_eval(%[[ZERO]] -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %[[ONE]] -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST-NEXT:   omp.teams {
// CIR-HOST-NEXT:     %[[IV_ADDR:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
// CIR-HOST-NEXT:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST-NEXT:     omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:       omp.loop_nest (%[[IV:.*]]) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[AONE]]) {
// CIR-HOST-NEXT:     %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-HOST-NEXT:     cir.store align(4) %[[IV_CIR]], %[[IV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.call @{{.*}}during
// CIR-HOST:          omp.yield
// CIR-HOST:        }
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@explicit_target_teams_distribute(
// CIR-DEVICE-NEXT: omp.target kernel_type(generic) {
// CIR-DEVICE-NEXT:   omp.teams {
// CIR-DEVICE-NEXT:     %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-NEXT:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-NEXT:     %[[I_ADDR:.*]] = cir.cast address_space %[[I_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// CIR-DEVICE-NEXT:     %[[IV_ADDR:.*]] = cir.cast address_space %[[IV_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// The loop only runs if its precondition `0 < 10` holds.
// CIR-DEVICE-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The device has no host_eval operands, so it recomputes the same trip count
// inside the region.
// CIR-DEVICE-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-DEVICE-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-DEVICE-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-DEVICE-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-DEVICE-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-DEVICE-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-DEVICE-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-DEVICE-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-DEVICE-NEXT:     omp.distribute private(@{{.*}} %[[I_ADDR]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:       omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
// CIR-DEVICE-NEXT:     %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-DEVICE-NEXT:     cir.store align(4) %[[IV_CIR]], %[[IV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.call @{{.*}}during
// CIR-DEVICE:          omp.yield
// CIR-DEVICE:        }
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

// The combined `target teams distribute` form decomposes into the same
// target/teams/distribute directive nesting as above, so it lowers
// identically.
void target_teams_distribute() {
#pragma omp target teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_distribute(
// The loop only runs if its precondition `0 < 10` holds.
// CIR-HOST-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The host computes the trip count before the omp.target and passes it
// in through host_eval.
// tripCount = (10 - (0 - 1 + 1)) / 1 = 10
// CIR-HOST-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-HOST-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-HOST-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-HOST-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-HOST-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-HOST-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-HOST-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: omp.target kernel_type(generic) host_eval(%[[ZERO]] -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %[[ONE]] -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST-NEXT:   omp.teams {
// CIR-HOST-NEXT:     %[[IV_ADDR:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
// CIR-HOST-NEXT:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST-NEXT:     omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:       omp.loop_nest (%[[IV:.*]]) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[AONE]]) {
// CIR-HOST-NEXT:     %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-HOST-NEXT:     cir.store align(4) %[[IV_CIR]], %[[IV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.call @{{.*}}during
// CIR-HOST:          omp.yield
// CIR-HOST:        }
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_distribute(
// CIR-DEVICE-NEXT: omp.target kernel_type(generic) {
// CIR-DEVICE-NEXT:   omp.teams {
// CIR-DEVICE-NEXT:     %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-NEXT:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-NEXT:     %[[I_ADDR:.*]] = cir.cast address_space %[[I_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// CIR-DEVICE-NEXT:     %[[IV_ADDR:.*]] = cir.cast address_space %[[IV_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// The loop only runs if its precondition `0 < 10` holds.
// CIR-DEVICE-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The device has no host_eval operands, so it recomputes the same trip count
// inside the region.
// CIR-DEVICE-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-DEVICE-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-DEVICE-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-DEVICE-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-DEVICE-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-DEVICE-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-DEVICE-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-DEVICE-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-DEVICE-NEXT:     omp.distribute private(@{{.*}} %[[I_ADDR]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:       omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
// CIR-DEVICE-NEXT:     %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-DEVICE-NEXT:     cir.store align(4) %[[IV_CIR]], %[[IV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.call @{{.*}}during
// CIR-DEVICE:          omp.yield
// CIR-DEVICE:        }
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

void target_teams_split_distribute() {
#pragma omp target teams
#pragma omp distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_split_distribute(
// The loop only runs if its precondition `0 < 10` holds.
// CIR-HOST-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The host computes the trip count before the omp.target and passes it
// in through host_eval.
// tripCount = (10 - (0 - 1 + 1)) / 1 = 10
// CIR-HOST-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-HOST-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-HOST-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-HOST-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-HOST-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-HOST-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-HOST-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: omp.target kernel_type(generic) host_eval(%[[ZERO]] -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %[[ONE]] -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST-NEXT:   omp.teams {
// CIR-HOST-NEXT:     %[[IV_ADDR:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
// CIR-HOST-NEXT:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST-NEXT:     omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:       omp.loop_nest (%[[IV:.*]]) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[AONE]]) {
// CIR-HOST-NEXT:     %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-HOST-NEXT:     cir.store align(4) %[[IV_CIR]], %[[IV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.call @{{.*}}during
// CIR-HOST:          omp.yield
// CIR-HOST:        }
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_split_distribute(
// CIR-DEVICE-NEXT: omp.target kernel_type(generic) {
// CIR-DEVICE-NEXT:   omp.teams {
// CIR-DEVICE-NEXT:     %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-NEXT:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-NEXT:     %[[I_ADDR:.*]] = cir.cast address_space %[[I_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// CIR-DEVICE-NEXT:     %[[IV_ADDR:.*]] = cir.cast address_space %[[IV_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// The loop only runs if its precondition `0 < 10` holds.
// CIR-DEVICE-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The device has no host_eval operands, so it recomputes the same trip count
// inside the region.
// CIR-DEVICE-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-DEVICE-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-DEVICE-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-DEVICE-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-DEVICE-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-DEVICE-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-DEVICE-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-DEVICE-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-DEVICE-NEXT:     omp.distribute private(@{{.*}} %[[I_ADDR]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:       omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
// CIR-DEVICE-NEXT:     %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-DEVICE-NEXT:     cir.store align(4) %[[IV_CIR]], %[[IV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.call @{{.*}}during
// CIR-DEVICE:          omp.yield
// CIR-DEVICE:        }
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

// A split `target`, with a separately, explicitly nested but combined
// `teams distribute`, is the fourth and final form: the other mix of
// combined/split not yet covered above. Just like the other three
// forms, its omp.target still needs its trip count host_eval'd, even
// though the nested construct is now a single `teams_distribute` AST node
// rather than two separate `teams`/`distribute` nodes.
void target_split_teams_distribute() {
#pragma omp target
#pragma omp teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_split_teams_distribute(
// The loop only runs if its precondition `0 < 10` holds.
// CIR-HOST-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The host computes the trip count before the omp.target and passes it
// in through host_eval.
// tripCount = (10 - (0 - 1 + 1)) / 1 = 10
// CIR-HOST-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-HOST-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-HOST-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-HOST-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-HOST-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-HOST-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-HOST-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: omp.target kernel_type(generic) host_eval(%[[ZERO]] -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %[[ONE]] -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST-NEXT:   omp.teams {
// CIR-HOST-NEXT:     %[[IV_ADDR:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
// CIR-HOST-NEXT:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST-NEXT:     omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:       omp.loop_nest (%[[IV:.*]]) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[AONE]]) {
// CIR-HOST-NEXT:     %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-HOST-NEXT:     cir.store align(4) %[[IV_CIR]], %[[IV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.call @{{.*}}during
// CIR-HOST:          omp.yield
// CIR-HOST:        }
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_split_teams_distribute(
// CIR-DEVICE-NEXT: omp.target kernel_type(generic) {
// CIR-DEVICE-NEXT:   omp.teams {
// CIR-DEVICE-NEXT:     %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-NEXT:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-NEXT:     %[[I_ADDR:.*]] = cir.cast address_space %[[I_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// CIR-DEVICE-NEXT:     %[[IV_ADDR:.*]] = cir.cast address_space %[[IV_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// The loop only runs if its precondition `0 < 10` holds.
// CIR-DEVICE-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The device has no host_eval operands, so it recomputes the same trip count
// inside the region.
// CIR-DEVICE-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-DEVICE-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-DEVICE-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-DEVICE-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-DEVICE-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-DEVICE-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-DEVICE-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-DEVICE-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-DEVICE-NEXT:     omp.distribute private(@{{.*}} %[[I_ADDR]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:       omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
// CIR-DEVICE-NEXT:     %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-DEVICE-NEXT:     cir.store align(4) %[[IV_CIR]], %[[IV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.call @{{.*}}during
// CIR-DEVICE:          omp.yield
// CIR-DEVICE:        }
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

// Host wrappers: both paths store the trip count (10) in the kernel
// arguments. CIR groups all its helper functions together after the host
// wrappers, since the MLIR-to-LLVM translation emits them during the
// lowering pass itself; OGCG's direct AST walk emits each one right after
// its host wrapper. Only the host wrappers' positions differ, so the later
// three are checked separately for each path.
// LLVM-HOST-LABEL: define {{.*}} void @explicit_target_teams_distribute(
// LLVM-HOST:         %[[KARGS:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST:         %[[TRIPCOUNT_GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS]], i32 0, i32 8
// LLVM-HOST-NEXT:    store i64 10, ptr %[[TRIPCOUNT_GEP]], align 8
// LLVM-HOST:         call i32 @__tgt_target_kernel(
// LLVM-HOST:         br i1 %{{.*}}, label %[[FAILED:[^,]+]], label %{{.*}}
// LLVM-HOST:       {{^}}[[FAILED]]:
// LLVM-HOST-NEXT:    call void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_explicit_target_teams_distribute_l{{[0-9]+}}(ptr null)

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_teams_distribute(
// LLVM-HOST-CIR:         %[[KARGS2:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS2]], i32 0, i32 8
// LLVM-HOST-CIR-NEXT:    store i64 10, ptr %[[TRIPCOUNT_GEP2]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:         br i1 %{{.*}}, label %[[FAILED2:[^,]+]], label %{{.*}}
// LLVM-HOST-CIR:       {{^}}[[FAILED2]]:
// LLVM-HOST-CIR-NEXT:    call void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_teams_distribute_l{{[0-9]+}}(ptr null)

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_teams_split_distribute(
// LLVM-HOST-CIR:         %[[KARGS3:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP3:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS3]], i32 0, i32 8
// LLVM-HOST-CIR-NEXT:    store i64 10, ptr %[[TRIPCOUNT_GEP3]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:         br i1 %{{.*}}, label %[[FAILED3:[^,]+]], label %{{.*}}
// LLVM-HOST-CIR:       {{^}}[[FAILED3]]:
// LLVM-HOST-CIR-NEXT:    call void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_teams_split_distribute_l{{[0-9]+}}(ptr null)

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_split_teams_distribute(
// LLVM-HOST-CIR:         %[[KARGS4:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP4:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS4]], i32 0, i32 8
// LLVM-HOST-CIR-NEXT:    store i64 10, ptr %[[TRIPCOUNT_GEP4]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:         br i1 %{{.*}}, label %[[FAILED4:[^,]+]], label %{{.*}}
// LLVM-HOST-CIR:       {{^}}[[FAILED4]]:
// LLVM-HOST-CIR-NEXT:    call void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_split_teams_distribute_l{{[0-9]+}}(ptr null)

// Outlined target-entry functions: fork the teams region via
// __kmpc_fork_teams, same as the standalone `teams distribute` case in
// teams-distribute.c. The forked function runs the distribute loop with
// schedule 92 (static distribute); CIR and OGCG use a different runtime
// API and control-flow shape for it.
// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_explicit_target_teams_distribute_l{{[0-9]+}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[OUTLINED:[^)]+]])

// LLVM-HOST-CIR:       define internal void @[[OUTLINED]](
// LLVM-HOST-CIR:         store i32 0, ptr %[[LB:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 9, ptr %[[UB:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 1, ptr %[[STRIDE:.*]], align 4
// LLVM-HOST-CIR-NEXT:    %[[TID:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 92,
// LLVM-HOST-CIR-SAME:      ptr %{{[^,]+}}, ptr %[[LB]], ptr %[[UB]], ptr %[[STRIDE]],
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %[[IV:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT:[^,]+]], %{{.*}} ]
// LLVM-HOST-CIR:         %[[LOOP_CMP:.*]] = icmp ult i32 %[[IV]], %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %[[LOOP_CMP]], label %[[LOOP_BODY:[^,]+]], label %[[LOOP_EXIT:[^ ,]+]]
// LLVM-HOST-CIR:       {{^}}[[LOOP_EXIT]]:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-HOST-CIR:       {{^}}[[LOOP_BODY]]:
// LLVM-HOST-CIR:         call void @during(i32
// LLVM-HOST-CIR:         %[[NEXT]] = add nuw i32 %[[IV]], 1

// LLVM-HOST-OGCG:       define internal void @[[OUTLINED]](
// LLVM-HOST-OGCG:         store i32 0, ptr %[[OMP_LB:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST:.*]], align 4
// LLVM-HOST-OGCG:         %[[TID_OGCG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID_OGCG]], i32 92,
// LLVM-HOST-OGCG-SAME:      ptr %[[OMP_IS_LAST]], ptr %[[OMP_LB]], ptr %[[OMP_UB]], ptr %[[OMP_STRIDE]],
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID_OGCG]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_teams_distribute(
// LLVM-HOST-OGCG:         %[[KARGS2:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS2]], i32 0, i32 8
// LLVM-HOST-OGCG-NEXT:    store i64 10, ptr %[[TRIPCOUNT_GEP2]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:         br i1 %{{.*}}, label %[[FAILED2:[^,]+]], label %{{.*}}
// LLVM-HOST-OGCG:       {{^}}[[FAILED2]]:
// LLVM-HOST-OGCG-NEXT:    call void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_teams_distribute_l{{[0-9]+}}(ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_teams_distribute_l{{[0-9]+}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[OUTLINED2:[^)]+]])

// LLVM-HOST-CIR:       define internal void @[[OUTLINED2]](
// LLVM-HOST-CIR:         store i32 0, ptr %[[LB2:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 9, ptr %[[UB2:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 1, ptr %[[STRIDE2:.*]], align 4
// LLVM-HOST-CIR-NEXT:    %[[TID2:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 92,
// LLVM-HOST-CIR-SAME:      ptr %{{[^,]+}}, ptr %[[LB2]], ptr %[[UB2]], ptr %[[STRIDE2]],
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %[[IV2:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT2:[^,]+]], %{{.*}} ]
// LLVM-HOST-CIR:         %[[LOOP_CMP2:.*]] = icmp ult i32 %[[IV2]], %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %[[LOOP_CMP2]], label %[[LOOP_BODY2:[^,]+]], label %[[LOOP_EXIT2:[^ ,]+]]
// LLVM-HOST-CIR:       {{^}}[[LOOP_EXIT2]]:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-HOST-CIR:       {{^}}[[LOOP_BODY2]]:
// LLVM-HOST-CIR:         call void @during(i32
// LLVM-HOST-CIR:         %[[NEXT2]] = add nuw i32 %[[IV2]], 1

// LLVM-HOST-OGCG:       define internal void @[[OUTLINED2]](
// LLVM-HOST-OGCG:         store i32 0, ptr %[[OMP_LB2:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB2:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE2:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST2:.*]], align 4
// LLVM-HOST-OGCG:         %[[TID_OGCG2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID_OGCG2]], i32 92,
// LLVM-HOST-OGCG-SAME:      ptr %[[OMP_IS_LAST2]], ptr %[[OMP_LB2]], ptr %[[OMP_UB2]], ptr %[[OMP_STRIDE2]],
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID_OGCG2]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_teams_split_distribute(
// LLVM-HOST-OGCG:         %[[KARGS3:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP3:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS3]], i32 0, i32 8
// LLVM-HOST-OGCG-NEXT:    store i64 10, ptr %[[TRIPCOUNT_GEP3]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:         br i1 %{{.*}}, label %[[FAILED3:[^,]+]], label %{{.*}}
// LLVM-HOST-OGCG:       {{^}}[[FAILED3]]:
// LLVM-HOST-OGCG-NEXT:    call void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_teams_split_distribute_l{{[0-9]+}}(ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_teams_split_distribute_l{{[0-9]+}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[OUTLINED3:[^)]+]])

// LLVM-HOST-CIR:       define internal void @[[OUTLINED3]](
// LLVM-HOST-CIR:         store i32 0, ptr %[[LB3:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 9, ptr %[[UB3:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 1, ptr %[[STRIDE3:.*]], align 4
// LLVM-HOST-CIR-NEXT:    %[[TID3:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 92,
// LLVM-HOST-CIR-SAME:      ptr %{{[^,]+}}, ptr %[[LB3]], ptr %[[UB3]], ptr %[[STRIDE3]],
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %[[IV3:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT3:[^,]+]], %{{.*}} ]
// LLVM-HOST-CIR:         %[[LOOP_CMP3:.*]] = icmp ult i32 %[[IV3]], %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %[[LOOP_CMP3]], label %[[LOOP_BODY3:[^,]+]], label %[[LOOP_EXIT3:[^ ,]+]]
// LLVM-HOST-CIR:       {{^}}[[LOOP_EXIT3]]:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-HOST-CIR:       {{^}}[[LOOP_BODY3]]:
// LLVM-HOST-CIR:         call void @during(i32
// LLVM-HOST-CIR:         %[[NEXT3]] = add nuw i32 %[[IV3]], 1

// LLVM-HOST-OGCG:       define internal void @[[OUTLINED3]](
// LLVM-HOST-OGCG:         store i32 0, ptr %[[OMP_LB3:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB3:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE3:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST3:.*]], align 4
// LLVM-HOST-OGCG:         %[[TID_OGCG3:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID_OGCG3]], i32 92,
// LLVM-HOST-OGCG-SAME:      ptr %[[OMP_IS_LAST3]], ptr %[[OMP_LB3]], ptr %[[OMP_UB3]], ptr %[[OMP_STRIDE3]],
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID_OGCG3]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_split_teams_distribute(
// LLVM-HOST-OGCG:         %[[KARGS4:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP4:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS4]], i32 0, i32 8
// LLVM-HOST-OGCG-NEXT:    store i64 10, ptr %[[TRIPCOUNT_GEP4]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:         br i1 %{{.*}}, label %[[FAILED4:[^,]+]], label %{{.*}}
// LLVM-HOST-OGCG:       {{^}}[[FAILED4]]:
// LLVM-HOST-OGCG-NEXT:    call void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_split_teams_distribute_l{{[0-9]+}}(ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_split_teams_distribute_l{{[0-9]+}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[OUTLINED4:[^)]+]])

// LLVM-HOST-CIR:       define internal void @[[OUTLINED4]](
// LLVM-HOST-CIR:         store i32 0, ptr %[[LB4:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 9, ptr %[[UB4:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 1, ptr %[[STRIDE4:.*]], align 4
// LLVM-HOST-CIR-NEXT:    %[[TID4:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID4]], i32 92,
// LLVM-HOST-CIR-SAME:      ptr %{{[^,]+}}, ptr %[[LB4]], ptr %[[UB4]], ptr %[[STRIDE4]],
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %[[IV4:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT4:[^,]+]], %{{.*}} ]
// LLVM-HOST-CIR:         %[[LOOP_CMP4:.*]] = icmp ult i32 %[[IV4]], %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %[[LOOP_CMP4]], label %[[LOOP_BODY4:[^,]+]], label %[[LOOP_EXIT4:[^ ,]+]]
// LLVM-HOST-CIR:       {{^}}[[LOOP_EXIT4]]:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID4]])
// LLVM-HOST-CIR:       {{^}}[[LOOP_BODY4]]:
// LLVM-HOST-CIR:         call void @during(i32
// LLVM-HOST-CIR:         %[[NEXT4]] = add nuw i32 %[[IV4]], 1

// LLVM-HOST-OGCG:       define internal void @[[OUTLINED4]](
// LLVM-HOST-OGCG:         store i32 0, ptr %[[OMP_LB4:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB4:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE4:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST4:.*]], align 4
// LLVM-HOST-OGCG:         %[[TID_OGCG4:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID_OGCG4]], i32 92,
// LLVM-HOST-OGCG-SAME:      ptr %[[OMP_IS_LAST4]], ptr %[[OMP_LB4]], ptr %[[OMP_UB4]], ptr %[[OMP_STRIDE4]],
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID_OGCG4]])
// LLVM-HOST-OGCG-NEXT:    ret void

// Device kernel entry: __kmpc_target_init/__kmpc_target_deinit and the
// branch into the user code are identical on both paths.
//
// CIR's device-side `distribute` lowering uses the loop-callback-based
// __kmpc_distribute_static_loop_4u API, which takes the trip count and needs
// an extra level of outlining for the per-iteration callback (teams ->
// distribute -> loop body); OGCG's distribute lowering uses the same direct
// init/fini API as on the host, needing only one level of outlining (teams
// -> loop body).
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_explicit_target_teams_distribute_l{{[0-9]+}}(
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:         br i1 %{{.*}}, label %[[USER_CODE:[^,]+]], label %{{.*}}
// LLVM-DEVICE:       {{^}}[[USER_CODE]]:
// LLVM-DEVICE:         call void @[[DEV_TEAMS:__omp_offloading_[^(]+]](
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE-NEXT:    ret void

// LLVM-DEVICE-CIR:       define internal void @[[DEV_TEAMS]](
// LLVM-DEVICE-CIR:         call void @[[DEV_DIST:__omp_offloading_[^(]+]](
// LLVM-DEVICE-CIR:       define internal void @[[DEV_DIST]](
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_static_loop_4u(ptr {{.*}}, ptr @[[DEV_BODY:[^,]+]], ptr {{.*}}, i32 10, i32 0, i8 0)
// LLVM-DEVICE-CIR:       define internal void @[[DEV_BODY]](
// LLVM-DEVICE-CIR:         call void @during(i32

// LLVM-DEVICE-OGCG:       define internal void @[[DEV_TEAMS]](
// LLVM-DEVICE-OGCG:         store i32 0, ptr %[[DEV_LB:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 9, ptr %[[DEV_UB:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 1, ptr %[[DEV_STRIDE:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 0, ptr %[[DEV_IS_LAST:.*]], align 4
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(
// LLVM-DEVICE-OGCG-SAME:      ptr {{.*}}, i32 %{{.*}}, i32 92,
// LLVM-DEVICE-OGCG-SAME:      ptr %[[DEV_IS_LAST]], ptr %[[DEV_LB]], ptr %[[DEV_UB]], ptr %[[DEV_STRIDE]],
// LLVM-DEVICE-OGCG-SAME:      i32 1, i32 1)
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_teams_distribute_l{{[0-9]+}}(
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:         br i1 %{{.*}}, label %[[USER_CODE2:[^,]+]], label %{{.*}}
// LLVM-DEVICE:       {{^}}[[USER_CODE2]]:
// LLVM-DEVICE:         call void @[[DEV_TEAMS2:__omp_offloading_[^(]+]](
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE-NEXT:    ret void

// LLVM-DEVICE-CIR:       define internal void @[[DEV_TEAMS2]](
// LLVM-DEVICE-CIR:         call void @[[DEV_DIST2:__omp_offloading_[^(]+]](
// LLVM-DEVICE-CIR:       define internal void @[[DEV_DIST2]](
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_static_loop_4u(ptr {{.*}}, ptr @[[DEV_BODY2:[^,]+]], ptr {{.*}}, i32 10, i32 0, i8 0)
// LLVM-DEVICE-CIR:       define internal void @[[DEV_BODY2]](
// LLVM-DEVICE-CIR:         call void @during(i32

// LLVM-DEVICE-OGCG:       define internal void @[[DEV_TEAMS2]](
// LLVM-DEVICE-OGCG:         store i32 0, ptr %[[DEV_LB2:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 9, ptr %[[DEV_UB2:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 1, ptr %[[DEV_STRIDE2:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 0, ptr %[[DEV_IS_LAST2:.*]], align 4
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(
// LLVM-DEVICE-OGCG-SAME:      ptr {{.*}}, i32 %{{.*}}, i32 92,
// LLVM-DEVICE-OGCG-SAME:      ptr %[[DEV_IS_LAST2]], ptr %[[DEV_LB2]], ptr %[[DEV_UB2]], ptr %[[DEV_STRIDE2]],
// LLVM-DEVICE-OGCG-SAME:      i32 1, i32 1)
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_teams_split_distribute_l{{[0-9]+}}(
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:         br i1 %{{.*}}, label %[[USER_CODE3:[^,]+]], label %{{.*}}
// LLVM-DEVICE:       {{^}}[[USER_CODE3]]:
// LLVM-DEVICE:         call void @[[DEV_TEAMS3:__omp_offloading_[^(]+]](
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE-NEXT:    ret void

// LLVM-DEVICE-CIR:       define internal void @[[DEV_TEAMS3]](
// LLVM-DEVICE-CIR:         call void @[[DEV_DIST3:__omp_offloading_[^(]+]](
// LLVM-DEVICE-CIR:       define internal void @[[DEV_DIST3]](
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_static_loop_4u(ptr {{.*}}, ptr @[[DEV_BODY3:[^,]+]], ptr {{.*}}, i32 10, i32 0, i8 0)
// LLVM-DEVICE-CIR:       define internal void @[[DEV_BODY3]](
// LLVM-DEVICE-CIR:         call void @during(i32

// LLVM-DEVICE-OGCG:       define internal void @[[DEV_TEAMS3]](
// LLVM-DEVICE-OGCG:         store i32 0, ptr %[[DEV_LB3:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 9, ptr %[[DEV_UB3:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 1, ptr %[[DEV_STRIDE3:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 0, ptr %[[DEV_IS_LAST3:.*]], align 4
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(
// LLVM-DEVICE-OGCG-SAME:      ptr {{.*}}, i32 %{{.*}}, i32 92,
// LLVM-DEVICE-OGCG-SAME:      ptr %[[DEV_IS_LAST3]], ptr %[[DEV_LB3]], ptr %[[DEV_UB3]], ptr %[[DEV_STRIDE3]],
// LLVM-DEVICE-OGCG-SAME:      i32 1, i32 1)
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{[0-9a-f]+_[0-9a-f]+}}_target_split_teams_distribute_l{{[0-9]+}}(
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:         br i1 %{{.*}}, label %[[USER_CODE4:[^,]+]], label %{{.*}}
// LLVM-DEVICE:       {{^}}[[USER_CODE4]]:
// LLVM-DEVICE:         call void @[[DEV_TEAMS4:__omp_offloading_[^(]+]](
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE-NEXT:    ret void

// LLVM-DEVICE-CIR:       define internal void @[[DEV_TEAMS4]](
// LLVM-DEVICE-CIR:         call void @[[DEV_DIST4:__omp_offloading_[^(]+]](
// LLVM-DEVICE-CIR:       define internal void @[[DEV_DIST4]](
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_static_loop_4u(ptr {{.*}}, ptr @[[DEV_BODY4:[^,]+]], ptr {{.*}}, i32 10, i32 0, i8 0)
// LLVM-DEVICE-CIR:       define internal void @[[DEV_BODY4]](
// LLVM-DEVICE-CIR:         call void @during(i32

// LLVM-DEVICE-OGCG:       define internal void @[[DEV_TEAMS4]](
// LLVM-DEVICE-OGCG:         store i32 0, ptr %[[DEV_LB4:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 9, ptr %[[DEV_UB4:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 1, ptr %[[DEV_STRIDE4:.*]], align 4
// LLVM-DEVICE-OGCG-NEXT:    store i32 0, ptr %[[DEV_IS_LAST4:.*]], align 4
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(
// LLVM-DEVICE-OGCG-SAME:      ptr {{.*}}, i32 %{{.*}}, i32 92,
// LLVM-DEVICE-OGCG-SAME:      ptr %[[DEV_IS_LAST4]], ptr %[[DEV_LB4]], ptr %[[DEV_UB4]], ptr %[[DEV_STRIDE4]],
// LLVM-DEVICE-OGCG-SAME:      i32 1, i32 1)
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(
