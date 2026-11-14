// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR

// Lowering to LLVM IR: the fork_call wrapper is identical on both paths.
// The worksharing-loop body itself uses a different runtime API and
// control-flow shape on each path.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-OGCG

void during(int);

void parallel_for() {
  // CIR: cir.func{{.*}}@{{.*}}parallel_for
#pragma omp parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }

  // CIR: omp.parallel {

  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>

  // The loop only runs if its precondition `0 < 10` holds.
  // CIR: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
  // The loop bounds are normalized to a `[0, tripCount)` range; tripCount is
  // computed from the original bounds/step.
  // CIR-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
  // CIR-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
  // CIR-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
  // CIR-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
  // CIR-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32

  // CIR: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {

  // CIR: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.call @{{.*}}during

  // CIR: omp.yield
  // CIR: }
  // CIR: }
  // CIR: omp.terminator
  // The parallel is a non-innermost leaf of the combined construct.
  // CIR: } {omp.combined}
}

// LLVM-LABEL: define {{.*}} void @parallel_for()
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @[[OUTLINED:[^)]+]])

// The worksharing loop itself: CIR uses the unsigned-trip-count
// __kmpc_for_static_init_4u/fini pair and synthesizes a canonical
// omp_loop.header/cond/body/inc/exit skeleton.
// OGCG uses the signed
// __kmpc_for_static_init_4/fini pair with a classic cond.true/false/end
// upper-bound clamp and an omp.inner.for.cond/body/inc skeleton.
// CIR also emits a trailing barrier after the loop that OGCG omits here,
// this is an optimization opportunity that the OMP MLIR lowering could
// also take advantage of in the future.
//
// LLVM-CIR:       define internal void @[[OUTLINED]](
// LLVM-CIR-SAME:    ptr noalias %{{.*}}, ptr noalias %{{.*}})
// LLVM-CIR:         %[[LASTITER:.*]] = alloca i32, align 4
// LLVM-CIR-NEXT:    %[[LB:.*]] = alloca i32, align 4
// LLVM-CIR-NEXT:    %[[UB:.*]] = alloca i32, align 4
// LLVM-CIR-NEXT:    %[[STRIDE:.*]] = alloca i32, align 4
// LLVM-CIR:         store i32 0, ptr %[[LB]], align 4
// LLVM-CIR-NEXT:    store i32 9, ptr %[[UB]], align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %[[STRIDE]], align 4
// LLVM-CIR:         %[[TID:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 34,
// LLVM-CIR-SAME:      ptr %[[LASTITER]], ptr %[[LB]], ptr %[[UB]], ptr %[[STRIDE]],
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %[[IV:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT:[^,]+]], %{{.*}} ]
// LLVM-CIR:         %[[LOOP_CMP:.*]] = icmp ult i32 %[[IV]], %{{.*}}
// LLVM-CIR-NEXT:    br i1 %[[LOOP_CMP]], label %[[LOOP_BODY:[^,]+]], label %[[LOOP_EXIT:[^ ,]+]]
// LLVM-CIR:       {{^}}[[LOOP_EXIT]]:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-CIR:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-CIR:       {{^}}[[LOOP_BODY]]:
// LLVM-CIR:         call void @during(i32
// LLVM-CIR:         %[[NEXT]] = add nuw i32 %[[IV]], 1
// LLVM-CIR:         ret void

// LLVM-OGCG:       define internal void @[[OUTLINED]](
// LLVM-OGCG-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^)]+}})
// LLVM-OGCG:         store i32 0, ptr %[[OMP_LB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST:.*]], align 4
// LLVM-OGCG:         %[[TID2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 34,
// LLVM-OGCG-SAME:      ptr %[[OMP_IS_LAST]], ptr %[[OMP_LB]], ptr %[[OMP_UB]], ptr %[[OMP_STRIDE]],
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[UB_VAL:.*]] = load i32, ptr %[[OMP_UB]], align 4
// LLVM-OGCG-NEXT:    %[[CMP:.*]] = icmp sgt i32 %[[UB_VAL]], 9
// LLVM-OGCG-NEXT:    br i1 %[[CMP]], label %[[COND_TRUE:[^,]+]], label %[[COND_FALSE:[^ ,]+]]
// LLVM-OGCG:         %[[COND:.*]] = phi i32 [ 9, %[[COND_TRUE]] ], [ %{{.*}}, %[[COND_FALSE]] ]
// LLVM-OGCG-NEXT:    store i32 %[[COND]], ptr %[[OMP_UB]], align 4
// LLVM-OGCG:         %[[CMP1:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP1]], label %[[FOR_BODY:[^,]+]], label %{{.*}}
// LLVM-OGCG:       {{^}}[[FOR_BODY]]:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-OGCG-NEXT:    ret void
