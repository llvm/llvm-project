// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void during(int);

// The combined `parallel for` directive decomposes into a `parallel` leaf and a
// `for` leaf. It lowers to an omp.wsloop + omp.loop_nest nested directly inside
// an omp.parallel, mirroring the separate `parallel` / `for` nesting.
void parallel_for() {
  // CHECK: cir.func{{.*}}@{{.*}}parallel_for
#pragma omp parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }

  // CHECK: omp.parallel {

  // The normalized 0-based counter and the real induction variable each get
  // their own alloca before the wsloop; "i"'s alloca has no `init` because
  // its value is now produced by the update expression below, not by
  // directly emitting the for-statement's init.
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>

  // The loop bounds are normalized to a `[0, tripCount)` range; tripCount is
  // computed from the original bounds/step via Sema's helper expression.
  // CHECK: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[LBM1:.*]] = cir.sub nsw %[[C0_CIR]], %[[C1_CIR]] : !s32i
  // CHECK: %[[C1B:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[LBM1P1:.*]] = cir.add nsw %[[LBM1]], %[[C1B]] : !s32i
  // CHECK: %[[SPAN:.*]] = cir.sub nsw %[[C10_CIR]], %[[LBM1P1]] : !s32i
  // CHECK: %[[C1C:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[C1C]] : !s32i
  // CHECK: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CHECK: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
  // CHECK: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32

  // "i" is registered as predetermined-private on the wsloop's `private`
  // clause: %[[I_ALLOCA]] (never read; the privatizer below has no init/copy
  // regions) is the "mold" operand, and the loop body uses the matching
  // block argument instead, giving each thread its own storage.
  // CHECK: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {

  // The normalized counter block argument is stored into the normalized
  // counter's alloca, then Sema's update expression (`i = 0 + 1 * iv`)
  // recomputes the real induction variable from it.
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // The parallel is a non-innermost leaf of the combined construct.
  // CHECK: } {omp.combined}
}
