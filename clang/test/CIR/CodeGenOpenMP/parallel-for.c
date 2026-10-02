// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR

// Lowering to LLVM IR: the fork_call wrapper is identical on both paths
// (only the forked function's naming suffix differs, `..omp_par` vs
// `.omp_outlined`
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

  // The loop bounds are normalized to a `[0, tripCount)` range; tripCount is
  // computed from the original bounds/step.
  // CIR: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CIR: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[LBM1:.*]] = cir.sub nsw %[[C0_CIR]], %[[C1_CIR]] : !s32i
  // CIR: %[[C1B:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[LBM1P1:.*]] = cir.add nsw %[[LBM1]], %[[C1B]] : !s32i
  // CIR: %[[SPAN:.*]] = cir.sub nsw %[[C10_CIR]], %[[LBM1P1]] : !s32i
  // CIR: %[[C1C:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[C1C]] : !s32i
  // CIR: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32

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
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @parallel_for{{.*}})

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
// LLVM-CIR-LABEL: define internal void @parallel_for..omp_par(
// LLVM-CIR-SAME:    ptr noalias %tid.addr, ptr noalias %zero.addr)
// LLVM-CIR:         %p.lastiter = alloca i32, align 4
// LLVM-CIR-NEXT:    %p.lowerbound = alloca i32, align 4
// LLVM-CIR-NEXT:    %p.upperbound = alloca i32, align 4
// LLVM-CIR-NEXT:    %p.stride = alloca i32, align 4
// LLVM-CIR:         store i32 0, ptr %p.lowerbound, align 4
// LLVM-CIR-NEXT:    store i32 9, ptr %p.upperbound, align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %p.stride, align 4
// LLVM-CIR:         %[[TID:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 34,
// LLVM-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-CIR:       omp_loop.exit:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-CIR:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-CIR:       omp_loop.body:
// LLVM-CIR:         call void @during(i32
// LLVM-CIR:         %omp_loop.next = add nuw i32 %omp_loop.iv, 1
// LLVM-CIR:         ret void

// LLVM-OGCG-LABEL: define internal void @parallel_for.omp_outlined(
// LLVM-OGCG-SAME:    ptr noalias noundef %.global_tid., ptr noalias noundef %.bound_tid.)
// LLVM-OGCG:         %.omp.lb = alloca i32, align 4
// LLVM-OGCG-NEXT:    %.omp.ub = alloca i32, align 4
// LLVM-OGCG-NEXT:    %.omp.stride = alloca i32, align 4
// LLVM-OGCG-NEXT:    %.omp.is_last = alloca i32, align 4
// LLVM-OGCG:         store i32 0, ptr %.omp.lb, align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %.omp.ub, align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// LLVM-OGCG:         %[[TID2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 34,
// LLVM-OGCG-SAME:      ptr %.omp.is_last, ptr %.omp.lb, ptr %.omp.ub, ptr %.omp.stride,
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[UB:.*]] = load i32, ptr %.omp.ub, align 4
// LLVM-OGCG-NEXT:    %[[CMP:.*]] = icmp sgt i32 %[[UB]], 9
// LLVM-OGCG-NEXT:    br i1 %[[CMP]], label %cond.true, label %cond.false
// LLVM-OGCG:       cond.end:
// LLVM-OGCG:         %[[COND:.*]] = phi i32 [ 9, %cond.true ], [ %{{.*}}, %cond.false ]
// LLVM-OGCG-NEXT:    store i32 %[[COND]], ptr %.omp.ub, align 4
// LLVM-OGCG:       omp.inner.for.cond:
// LLVM-OGCG:         %[[CMP1:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP1]], label %omp.inner.for.body, label %omp.inner.for.end
// LLVM-OGCG:       omp.inner.for.body:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:       omp.loop.exit:
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-OGCG-NEXT:    ret void
