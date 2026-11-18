// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR

void during(int);
int getLB(void);

// All the loop forms below normalize to an omp.loop_nest iterating a
// `[0, tripCount)` range, where tripCount is computed from the original
// bounds/step. With constant bounds, tripCount is computed as
// `(hi - (lo - step + 1)) / step`, where `hi`/`lo` are the larger/smaller
// bound and `step` is the step's magnitude.

// Decreasing loop: `i > 0; i--` (induction variable on the left, plain
// unary decrement).
void dec_unary() {
  // CIR: cir.func{{.*}}@{{.*}}dec_unary
#pragma omp for
  for (int i = 20; i > 0; i--) {
    during(i);
  }
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // The loop only runs if its precondition `20 > 0` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp gt %[[PRE_L]], %[[PRE_R]] : !s32i
  // tripCount = (20 - (0 - 1 + 1)) / 1 = 20
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE]] : !s32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
  // CIR: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CIR: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 20 - 1 * iv`.
  // CIR: %[[C20:.*]] = cir.const #cir.int<20> : !s32i
  // CIR: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[C1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C1]] : !s32i
  // CIR: %[[SUB:.*]] = cir.sub nsw %[[C20]], %[[MUL]] : !s32i
  // CIR: cir.store align(4) %[[SUB]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.call @{{.*}}during
}

// Decreasing loop using compound assignment: `i -= 2`.
void dec_compound() {
  // CIR: cir.func{{.*}}@{{.*}}dec_compound
#pragma omp for
  for (int i = 20; i > 0; i -= 2) {
    during(i);
  }
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // The loop only runs if its precondition `20 > 0` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp gt %[[PRE_L]], %[[PRE_R]] : !s32i
  // tripCount = (20 - (0 - 2 + 1)) / 2 = 10
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<2> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE]] : !s32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<2> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
  // CIR: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CIR: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 20 - 2 * iv`.
  // CIR: %[[C20:.*]] = cir.const #cir.int<20> : !s32i
  // CIR: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[C2:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C2]] : !s32i
  // CIR: %[[SUB:.*]] = cir.sub nsw %[[C20]], %[[MUL]] : !s32i
  // CIR: cir.store align(4) %[[SUB]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.call @{{.*}}during
}

// Decreasing loop using the `i = i - 2` canonical assignment form.
void dec_assign_sub() {
  // CIR: cir.func{{.*}}@{{.*}}dec_assign_sub
#pragma omp for
  for (int i = 20; i > 0; i = i - 2) {
    during(i);
  }
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // The loop only runs if its precondition `20 > 0` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp gt %[[PRE_L]], %[[PRE_R]] : !s32i
  // tripCount = (20 - (0 - 2 + 1)) / 2 = 10
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<2> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE]] : !s32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<2> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
  // CIR: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CIR: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 20 - 2 * iv`.
  // CIR: %[[C20:.*]] = cir.const #cir.int<20> : !s32i
  // CIR: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[C2:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C2]] : !s32i
  // CIR: %[[SUB:.*]] = cir.sub nsw %[[C20]], %[[MUL]] : !s32i
  // CIR: cir.store align(4) %[[SUB]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.call @{{.*}}during
}

// Increasing loop using the `i = 2 + i` canonical assignment form.
void inc_assign_add_rev() {
  // CIR: cir.func{{.*}}@{{.*}}inc_assign_add_rev
#pragma omp for
  for (int i = 0; i < 20; i = 2 + i) {
    during(i);
  }
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // The loop only runs if its precondition `0 < 20` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
  // tripCount = (20 - (0 - 2 + 1)) / 2 = 10
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<2> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE]] : !s32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<2> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
  // CIR: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CIR: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 0 + 2 * iv`.
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[C2:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C2]] : !s32i
  // CIR: %[[ADD:.*]] = cir.add nsw %[[C0]], %[[MUL]] : !s32i
  // CIR: cir.store align(4) %[[ADD]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.call @{{.*}}during
}

// Increasing loop with the upper bound on the left of the comparison
// (`20 > i`).
void ub_on_left() {
  // CIR: cir.func{{.*}}@{{.*}}ub_on_left
#pragma omp for
  for (int i = 0; 20 > i; i++) {
    during(i);
  }
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // The loop only runs if its precondition `0 < 20` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
  // tripCount = (20 - (0 - 1 + 1)) / 1 = 20
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<20> : !s32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE]] : !s32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
  // CIR: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CIR: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 0 + 1 * iv`.
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[C1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C1]] : !s32i
  // CIR: %[[ADD:.*]] = cir.add nsw %[[C0]], %[[MUL]] : !s32i
  // CIR: cir.store align(4) %[[ADD]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.call @{{.*}}during
}

// Loop with a side-effecting init expression (a function call, `getLB()`).
void side_effecting_init() {
  // CIR: cir.func{{.*}}@{{.*}}side_effecting_init
#pragma omp for
  for (int i = getLB(); i < 10; i++) {
    during(i);
  }
  // CIR: %[[CAPTURE:.*]] = cir.alloca ".capture_expr." align(4) init : !cir.ptr<!s32i>
  // CIR-NEXT: %[[LAST_ITER:.*]] = cir.alloca ".capture_expr." align(4) init : !cir.ptr<!u32i>
  // CIR: %[[CALL:.*]] = cir.call @{{.*}}getLB{{.*}}() : {{.*}} -> !s32i
  // CIR-NEXT: cir.store{{.*}} %[[CALL]], %[[CAPTURE]]
  // With a non-constant start, the computation is done in unsigned
  // arithmetic: lastIter = (10 - lb - 1 + 1) / 1 - 1 is captured, and
  // tripCount = lastIter + 1.
  // CIR-NEXT: %[[UB:.*]] = cir.const #cir.int<10> : !u32i
  // CIR-NEXT: %[[LB:.*]] = cir.load{{.*}} %[[CAPTURE]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT: %[[LB_U:.*]] = cir.cast integral %[[LB]] : !s32i -> !u32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub %[[UB]], %[[LB_U]] : !u32i
  // CIR-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !u32i
  // CIR-NEXT: %[[SPAN_M1:.*]] = cir.sub %[[SPAN]], %[[ONE_A]] : !u32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !u32i
  // CIR-NEXT: %[[SPAN_ADJ:.*]] = cir.add %[[SPAN_M1]], %[[STEP_A]] : !u32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !u32i
  // CIR-NEXT: %[[NUM_ITERS:.*]] = cir.div %[[SPAN_ADJ]], %[[STEP_B]] : !u32i
  // CIR-NEXT: %[[ONE_B:.*]] = cir.const #cir.int<1> : !u32i
  // CIR-NEXT: %[[LAST:.*]] = cir.sub %[[NUM_ITERS]], %[[ONE_B]] : !u32i
  // CIR-NEXT: cir.store{{.*}} %[[LAST]], %[[LAST_ITER]] : !u32i, !cir.ptr<!u32i>
  // The loop only runs if its precondition `lb < 10` holds.
  // CIR-NEXT: %[[PRE_LB:.*]] = cir.load{{.*}} %[[CAPTURE]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT: %[[PRE_UB:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_LB]], %[[PRE_UB]] : !s32i
  // CIR-NEXT: %[[LAST_RELOAD:.*]] = cir.load{{.*}} %[[LAST_ITER]] : !cir.ptr<!u32i>, !u32i
  // CIR-NEXT: %[[ONE_C:.*]] = cir.const #cir.int<1> : !u32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.add %[[LAST_RELOAD]], %[[ONE_C]] : !u32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !u32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !u32i, !u32i) -> !u32i
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !u32i -> i32
  // CIR-NOT: cir.call @{{.*}}getLB
  // CIR: omp.wsloop private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%[[TRIPCOUNT]]) step (%{{.*}}) {
  // CIR-NOT: cir.call @{{.*}}getLB
  // CIR: cir.call @{{.*}}during
}

// Loop with a pointer induction variable (`int *p`).
void pointer_induction_var() {
  // CIR: cir.func{{.*}}@{{.*}}pointer_induction_var
  int a[10];
#pragma omp for
  for (int *p = a; p < a + 10; ++p) {
    during(*p);
  }
  // CIR: %[[BEGIN:.*]] = cir.alloca ".capture_expr." align(8) init : !cir.ptr<!cir.ptr<!s32i>>
  // CIR-NEXT: %[[END:.*]] = cir.alloca ".capture_expr." align(8) init : !cir.ptr<!cir.ptr<!s32i>>
  // CIR-NEXT: %[[LAST_ITER:.*]] = cir.alloca ".capture_expr." align(8) init : !cir.ptr<!s64i>
  // CIR: %[[P_ALLOCA:.*]] = cir.alloca "p" align(8) : !cir.ptr<!cir.ptr<!s32i>>
  // The span is the pointer difference between the captured bounds:
  // lastIter = ((a + 10) - a - 1 + 1) / 1 - 1 is captured, and
  // tripCount = lastIter + 1.
  // CIR: %[[END_VAL:.*]] = cir.load{{.*}} %[[END]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR-NEXT: %[[BEGIN_VAL:.*]] = cir.load{{.*}} %[[BEGIN]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR-NEXT: %[[SPAN:.*]] = cir.ptr_diff %[[END_VAL]], %[[BEGIN_VAL]] : !cir.ptr<!s32i> -> !s64i
  // CIR-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s64i
  // CIR-NEXT: %[[SPAN_M1:.*]] = cir.sub nsw %[[SPAN]], %[[ONE_A]] : !s64i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s64i
  // CIR-NEXT: %[[SPAN_ADJ:.*]] = cir.add nsw %[[SPAN_M1]], %[[STEP_A]] : !s64i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s64i
  // CIR-NEXT: %[[NUM_ITERS:.*]] = cir.div %[[SPAN_ADJ]], %[[STEP_B]] : !s64i
  // CIR-NEXT: %[[ONE_B:.*]] = cir.const #cir.int<1> : !s64i
  // CIR-NEXT: %[[LAST:.*]] = cir.sub nsw %[[NUM_ITERS]], %[[ONE_B]] : !s64i
  // CIR-NEXT: cir.store{{.*}} %[[LAST]], %[[LAST_ITER]] : !s64i, !cir.ptr<!s64i>
  // The loop only runs if its precondition `a < a + 10` holds.
  // CIR-NEXT: %[[PRE_BEGIN:.*]] = cir.load{{.*}} %[[BEGIN]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR-NEXT: %[[PRE_END:.*]] = cir.load{{.*}} %[[END]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_BEGIN]], %[[PRE_END]] : !cir.ptr<!s32i>
  // CIR-NEXT: %[[LAST_RELOAD:.*]] = cir.load{{.*}} %[[LAST_ITER]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: %[[ONE_C:.*]] = cir.const #cir.int<1> : !s64i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.add nsw %[[LAST_RELOAD]], %[[ONE_C]] : !s64i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s64i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s64i, !s64i) -> !s64i
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s64i -> i64
  // CIR: omp.wsloop private(@{{.*}} %[[P_ALLOCA]] -> %[[P_PRIV:.*]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CIR-NEXT: omp.loop_nest (%{{.*}}) : i64 = (%{{.*}}) to (%[[TRIPCOUNT]]) step (%{{.*}}) {
  // CIR: %[[P_BASE:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: %[[P_NEXT:.*]] = cir.ptr_stride %[[P_BASE]], %{{.*}} : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[P_NEXT]], %[[P_PRIV]]
  // CIR: cir.call @{{.*}}during
}

// Pre-declared induction variable (plain assignment, no DeclStmt) with a
// `!=` condition.
void predeclared_var_and_not_equal_cond() {
  // CIR: cir.func{{.*}}@{{.*}}predeclared_var_and_not_equal_cond
  int i;
#pragma omp for
  for (i = 0; i != 10; i++) {
    during(i);
  }
  // CIR: cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_MOLD:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // The loop only runs if its precondition `0 < 10` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
  // A `!=` condition is treated like `<`.
  // tripCount = (10 - (0 - 1 + 1)) / 1 = 10
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE]] : !s32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
  // CIR: omp.wsloop private(@{{.*}} %[[I_MOLD]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%[[TRIPCOUNT]]) step (%{{.*}}) {
  // CIR: cir.store{{.*}} %{{.*}}, %[[IV_ALLOCA]]
  // CIR: cir.store{{.*}} %{{.*}}, %[[I_PRIV]]
  // CIR: cir.call @{{.*}}during
}

// Same pre-declared counter and `!=` condition, inside a combined
// `parallel for`.
void predeclared_var_inside_parallel_for() {
  // CIR: cir.func{{.*}}@{{.*}}predeclared_var_inside_parallel_for
  int i;
#pragma omp parallel for
  for (i = 0; i != 10; i++) {
    during(i);
  }
  // CIR: cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CIR: omp.parallel {
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_MOLD:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // The loop only runs if its precondition `0 < 10` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
  // tripCount = (10 - (0 - 1 + 1)) / 1 = 10
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE]] : !s32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
  // CIR: omp.wsloop private(@{{.*}} %[[I_MOLD]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%[[TRIPCOUNT]]) step (%{{.*}}) {
  // CIR: cir.store{{.*}} %{{.*}}, %[[IV_ALLOCA]]
  // CIR: cir.store{{.*}} %{{.*}}, %[[I_PRIV]]
  // CIR: cir.call @{{.*}}during
}

// Unsigned loop whose condition is false on entry. The unsigned iteration
// count arithmetic, (0 - (10 - 2 + 1)) / 2, wraps to a large positive value,
// so the trip count must come from the precondition `10 < 0` and be zero.
void unsigned_false_precond() {
  // CIR: cir.func{{.*}}@{{.*}}unsigned_false_precond
#pragma omp for
  for (unsigned i = 10; i < 0; i += 2) {
    during(i);
  }
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!u32i>
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<10> : !u32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<0> : !u32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !u32i
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<0> : !u32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<10> : !u32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<2> : !u32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub %[[LO]], %[[STEP_A]] : !u32i
  // CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add %[[LO_MINUS_STEP]], %[[ONE]] : !u32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub %[[HI]], %[[ADJ_LO]] : !u32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<2> : !u32i
  // CIR-NEXT: %[[NUM_ITERS:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !u32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !u32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[NUM_ITERS]] else %[[TC_ZERO]] : (!cir.bool, !u32i, !u32i) -> !u32i
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !u32i -> i32
  // CIR: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %{{.*}} : !cir.ptr<!u32i>) {
  // CIR-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%[[TRIPCOUNT]]) step (%{{.*}}) {
}
