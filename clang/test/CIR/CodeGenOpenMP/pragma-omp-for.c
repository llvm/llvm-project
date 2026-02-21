// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void before(int);
void during(int);
void after(int);

void emit_simple_for() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_simple_for
  int j = 5;
  before(j);
  // CHECK: cir.call @{{.*}}before
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
        during(j);
    }
  }
  // CHECK: omp.parallel {

  // CIR constants for bounds, then cast to std integer
  // CHECK: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i

  // induction variable alloca (emitted before wsloop)
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]

  // conversion to std integer for omp.loop_nest
  // CHECK: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[C0_CIR]] : !s32i to i32
  // CHECK: %[[C10:.*]] = builtin.unrealized_conversion_cast %[[C10_CIR]] : !s32i to i32
  // CHECK: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[C1_CIR]] : !s32i to i32

  // omp loop
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {

  // store induction variable block arg into alloca
  // CHECK: %[[IV_CIR:.*]] = builtin.unrealized_conversion_cast %[[IV]] : i32 to !s32i
  // CHECK: cir.store %[[IV_CIR]], %[[I_ALLOCA]] : !s32i, !cir.ptr<!s32i>

  // during(j)
  // CHECK: cir.load {{.*}} %{{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }
  after(j);
  // CHECK: cir.call @{{.*}}after
}

void emit_for_with_vars() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_with_vars
  int j = 5;
  before(j);
  // CHECK: cir.call @{{.*}}before
#pragma omp parallel
  {
    int lb = 1;
    long ub = 10;
    short step = 1;
#pragma omp for
    for (int i = 0; i < ub; i=i+step) {
        during(j);
    }
  }

  // CHECK: omp.parallel {

  // allocas
  // CHECK: %[[LB:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["lb", init]
  // CHECK: %[[UB:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["ub", init]
  // CHECK: %[[STEP:.*]] = cir.alloca !s16i, !cir.ptr<!s16i>, ["step", init]

  // stores
  // CHECK: cir.store {{.*}}, %[[LB]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.store {{.*}}, %[[UB]] : !s64i, !cir.ptr<!s64i>
  // CHECK: cir.store {{.*}}, %[[STEP]] : !s16i, !cir.ptr<!s16i>

  // lower bound (CIR constant + cast to i32)
  // CHECK: %[[LB0_CIR:.*]] = cir.const #cir.int<0> : !s32i

  // upper bound: load, integral cast to i32, then unrealized cast
  // CHECK: %[[UBLOAD:.*]] = cir.load {{.*}} %[[UB]] : !cir.ptr<!s64i>, !s64i
  // CHECK: %[[UBCAST:.*]] = cir.cast integral %[[UBLOAD]] : !s64i -> !s32i

  // step: load, integral cast to i32, then unrealized cast
  // CHECK: %[[STEPLOAD:.*]] = cir.load {{.*}} %[[STEP]] : !cir.ptr<!s16i>, !s16i
  // CHECK: %[[STEPCONV:.*]] = cir.cast integral %[[STEPLOAD]] : !s16i -> !s32i

  // induction variable alloca (emitted before wsloop)
  // CHECK: %[[I2_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]

  // conversion to std integer for omp.loop_nest
  // CHECK: %[[LB0:.*]] = builtin.unrealized_conversion_cast %[[LB0_CIR]] : !s32i to i32
  // CHECK: %[[UBSTD:.*]] = builtin.unrealized_conversion_cast %[[UBCAST]] : !s32i to i32
  // CHECK: %[[STEPSTD:.*]] = builtin.unrealized_conversion_cast %[[STEPCONV]] : !s32i to i32

  // omp loop
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%[[IV2:.*]]) : i32 = (%[[LB0]]) to (%[[UBSTD]]) step (%[[STEPSTD]]) {

  // store induction variable block arg into alloca
  // CHECK: %[[IV2_CIR:.*]] = builtin.unrealized_conversion_cast %[[IV2]] : i32 to !s32i
  // CHECK: cir.store %[[IV2_CIR]], %[[I2_ALLOCA]] : !s32i, !cir.ptr<!s32i>

  // during(j)
  // CHECK: cir.load {{.*}} %{{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }

  after(j);
  // CHECK: cir.call @{{.*}}after
}

void emit_for_with_induction_var() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_with_induction_var
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CIR constants
  // CHECK: %[[IC0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[IC10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[IC1_CIR:.*]] = cir.const #cir.int<1> : !s32i

  // induction variable alloca
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]

  // conversion to std integer
  // CHECK: %[[IC0:.*]] = builtin.unrealized_conversion_cast %[[IC0_CIR]] : !s32i to i32
  // CHECK: %[[IC10:.*]] = builtin.unrealized_conversion_cast %[[IC10_CIR]] : !s32i to i32
  // CHECK: %[[IC1:.*]] = builtin.unrealized_conversion_cast %[[IC1_CIR]] : !s32i to i32

  // omp loop
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%[[IV3:.*]]) : i32 = (%[[IC0]]) to (%[[IC10]]) step (%[[IC1]]) {

  // store induction variable into alloca
  // CHECK: %[[IV3_CIR:.*]] = builtin.unrealized_conversion_cast %[[IV3]] : i32 to !s32i
  // CHECK: cir.store %[[IV3_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>

  // during(i) - loads the induction variable from the alloca
  // CHECK: %[[I_VAL:.*]] = cir.load %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}during(%[[I_VAL]])

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }
}

// Test inclusive upper bound (i <= 9)
void emit_for_inclusive_bound() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_inclusive_bound
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i <= 9; i++) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: cir.const #cir.int<0> : !s32i
  // CHECK: cir.const #cir.int<9> : !s32i
  // CHECK: cir.const #cir.int<1> : !s32i
  // CHECK: %[[INC_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
  // CHECK: %[[INC_C0:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[INC_C9:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[INC_C1:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32

  // CHECK: omp.wsloop {
  // inclusive = true
  // CHECK-NEXT: omp.loop_nest (%[[INC_IV:.*]]) : i32 = (%[[INC_C0]]) to (%[[INC_C9]]) inclusive step (%[[INC_C1]]) {

  // CHECK: builtin.unrealized_conversion_cast %[[INC_IV]] : i32 to !s32i
  // CHECK: cir.store
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Test reversed comparison (10 > i)
void emit_for_reversed_cmp() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_reversed_cmp
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; 10 > i; i++) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: cir.const #cir.int<0> : !s32i
  // CHECK: cir.const #cir.int<10> : !s32i
  // CHECK: cir.const #cir.int<1> : !s32i
  // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
  // CHECK: %[[REV_C0:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[REV_C10:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[REV_C1:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[REV_C0]]) to (%[[REV_C10]]) step (%[[REV_C1]]) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Test reversed inclusive comparison (9 >= i)
void emit_for_reversed_inclusive_cmp() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_reversed_inclusive_cmp
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; 9 >= i; i++) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: cir.const #cir.int<0> : !s32i
  // CHECK: cir.const #cir.int<9> : !s32i
  // CHECK: cir.const #cir.int<1> : !s32i
  // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
  // CHECK: %[[RI_C0:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[RI_C9:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[RI_C1:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[RI_C0]]) to (%[[RI_C9]]) inclusive step (%[[RI_C1]]) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Test compound assignment step (i += 2)
void emit_for_compound_step() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_compound_step
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 20; i += 2) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: cir.const #cir.int<0> : !s32i
  // CHECK: cir.const #cir.int<20> : !s32i
  // CHECK: cir.const #cir.int<2> : !s32i
  // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
  // CHECK: %[[CS_C0:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[CS_C20:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[CS_C2:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[CS_C0]]) to (%[[CS_C20]]) step (%[[CS_C2]]) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Test commuted step expression (i = step + i)
void emit_for_commuted_step() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_commuted_step
  short step = 3;
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 30; i = step + i) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: cir.const #cir.int<0> : !s32i
  // CHECK: cir.const #cir.int<30> : !s32i

  // step is loaded and cast to the loop variable type (i32) in CIR
  // CHECK: %[[CM_STEP_LOAD:.*]] = cir.load {{.*}} : !cir.ptr<!s16i>, !s16i
  // CHECK: %[[CM_STEP_CIR:.*]] = cir.cast integral %[[CM_STEP_LOAD]] : !s16i -> !s32i

  // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]

  // conversion to std integer
  // CHECK: %[[CM_C0:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[CM_C30:.*]] = builtin.unrealized_conversion_cast {{.*}} : !s32i to i32
  // CHECK: %[[CM_STEP:.*]] = builtin.unrealized_conversion_cast %[[CM_STEP_CIR]] : !s32i to i32

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[CM_C0]]) to (%[[CM_C30]]) step (%[[CM_STEP]]) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}
