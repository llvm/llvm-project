// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void during(int);
int getLB(void);

// All the loop forms below normalize to an omp.loop_nest iterating a
// `[0, tripCount)` range, where tripCount is computed from the original
// bounds/step via Sema's helper expression (OMPLoopDirective::getNumIterations()),
// and the user's real induction variable "i" is recomputed each iteration
// from the normalized counter via Sema's update expression
// (OMPLoopDirective::updates()). This is what makes these forms "just work"
// without the CIR codegen having to pattern-match each one individually.

// Decreasing loop: `i > 0; i--` (induction variable on the left, plain
// unary decrement).
void dec_unary() {
  // CHECK: cir.func{{.*}}@{{.*}}dec_unary
#pragma omp for
  for (int i = 20; i > 0; i--) {
    during(i);
  }
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
  // CHECK: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CHECK: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CHECK: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32
  // CHECK: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 20 - 1 * iv`.
  // CHECK: %[[C20:.*]] = cir.const #cir.int<20> : !s32i
  // CHECK: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[C1:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C1]] : !s32i
  // CHECK: %[[SUB:.*]] = cir.sub nsw %[[C20]], %[[MUL]] : !s32i
  // CHECK: cir.store align(4) %[[SUB]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during
}

// Decreasing loop using compound assignment: `i -= 2`.
void dec_compound() {
  // CHECK: cir.func{{.*}}@{{.*}}dec_compound
#pragma omp for
  for (int i = 20; i > 0; i -= 2) {
    during(i);
  }
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
  // CHECK: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CHECK: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CHECK: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32
  // CHECK: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 20 - 2 * iv`.
  // CHECK: %[[C20:.*]] = cir.const #cir.int<20> : !s32i
  // CHECK: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[C2:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C2]] : !s32i
  // CHECK: %[[SUB:.*]] = cir.sub nsw %[[C20]], %[[MUL]] : !s32i
  // CHECK: cir.store align(4) %[[SUB]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during
}

// Decreasing loop using the `i = i - 2` canonical assignment form.
void dec_assign_sub() {
  // CHECK: cir.func{{.*}}@{{.*}}dec_assign_sub
#pragma omp for
  for (int i = 20; i > 0; i = i - 2) {
    during(i);
  }
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
  // CHECK: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CHECK: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CHECK: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32
  // CHECK: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 20 - 2 * iv`.
  // CHECK: %[[C20:.*]] = cir.const #cir.int<20> : !s32i
  // CHECK: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[C2:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C2]] : !s32i
  // CHECK: %[[SUB:.*]] = cir.sub nsw %[[C20]], %[[MUL]] : !s32i
  // CHECK: cir.store align(4) %[[SUB]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during
}

// Increasing loop using the `i = 2 + i` canonical assignment form.
void inc_assign_add_rev() {
  // CHECK: cir.func{{.*}}@{{.*}}inc_assign_add_rev
#pragma omp for
  for (int i = 0; i < 20; i = 2 + i) {
    during(i);
  }
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
  // CHECK: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CHECK: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CHECK: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32
  // CHECK: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 0 + 2 * iv`.
  // CHECK: %[[C0:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[C2:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C2]] : !s32i
  // CHECK: %[[ADD:.*]] = cir.add nsw %[[C0]], %[[MUL]] : !s32i
  // CHECK: cir.store align(4) %[[ADD]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during
}

// Increasing loop with the upper bound on the left of the comparison
// (`20 > i`).
void ub_on_left() {
  // CHECK: cir.func{{.*}}@{{.*}}ub_on_left
#pragma omp for
  for (int i = 0; 20 > i; i++) {
    during(i);
  }
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
  // CHECK: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CHECK: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CHECK: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32
  // CHECK: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // The update expression recomputes `i = 0 + 1 * iv`.
  // CHECK: %[[C0:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[IV_RELOAD:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[C1:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[MUL:.*]] = cir.mul nsw %[[IV_RELOAD]], %[[C1]] : !s32i
  // CHECK: %[[ADD:.*]] = cir.add nsw %[[C0]], %[[MUL]] : !s32i
  // CHECK: cir.store align(4) %[[ADD]], %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during
}

// Loop with a side-effecting init expression (a function call, `getLB()`).
void side_effecting_init() {
  // CHECK: cir.func{{.*}}@{{.*}}side_effecting_init
#pragma omp for
  for (int i = getLB(); i < 10; i++) {
    during(i);
  }
  // CHECK: %[[CAPTURE:.*]] = cir.alloca ".capture_expr." align(4) init : !cir.ptr<!s32i>
  // CHECK: %[[CALL:.*]] = cir.call @{{.*}}getLB{{.*}}() : {{.*}} -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[CALL]], %[[CAPTURE]]
  // CHECK-NOT: cir.call @{{.*}}getLB
  // CHECK: omp.wsloop private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
  // CHECK: omp.loop_nest
  // CHECK-NOT: cir.call @{{.*}}getLB
  // CHECK: cir.call @{{.*}}during
}

// Loop with a pointer induction variable (`int *p`).
void pointer_induction_var() {
  // CHECK: cir.func{{.*}}@{{.*}}pointer_induction_var
  int a[10];
#pragma omp for
  for (int *p = a; p < a + 10; ++p) {
    during(*p);
  }
  // CHECK: %[[P_ALLOCA:.*]] = cir.alloca "p" align(8) : !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: omp.wsloop private(@{{.*}} %[[P_ALLOCA]] -> %[[P_PRIV:.*]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i64 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK: %[[P_BASE:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: %[[P_NEXT:.*]] = cir.ptr_stride %[[P_BASE]], %{{.*}} : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
  // CHECK: cir.store{{.*}} %[[P_NEXT]], %[[P_PRIV]]
  // CHECK: cir.call @{{.*}}during
}

// Pre-declared induction variable (plain assignment, no DeclStmt) with a
// `!=` condition.
void predeclared_var_and_not_equal_cond() {
  // CHECK: cir.func{{.*}}@{{.*}}predeclared_var_and_not_equal_cond
  int i;
#pragma omp for
  for (i = 0; i != 10; i++) {
    during(i);
  }
  // CHECK: cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_MOLD:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: omp.wsloop private(@{{.*}} %[[I_MOLD]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK: cir.store{{.*}} %{{.*}}, %[[IV_ALLOCA]]
  // CHECK: cir.store{{.*}} %{{.*}}, %[[I_PRIV]]
  // CHECK: cir.call @{{.*}}during
}

// Same pre-declared counter and `!=` condition, inside a combined
// `parallel for`.
void predeclared_var_inside_parallel_for() {
  // CHECK: cir.func{{.*}}@{{.*}}predeclared_var_inside_parallel_for
  int i;
#pragma omp parallel for
  for (i = 0; i != 10; i++) {
    during(i);
  }
  // CHECK: cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: omp.parallel {
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_MOLD:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CHECK: omp.wsloop private(@{{.*}} %[[I_MOLD]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK: cir.store{{.*}} %{{.*}}, %[[IV_ALLOCA]]
  // CHECK: cir.store{{.*}} %{{.*}}, %[[I_PRIV]]
  // CHECK: cir.call @{{.*}}during
}
