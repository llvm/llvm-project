// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR

// Lowering to LLVM IR: mostly identical between CIR and OGCG, modulo how
// each path schedules the distribute and for chunking (see the comments
// below); they also just name the outlined functions differently.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-OGCG

void during(int);

// `distribute parallel for` is a composite construct. It lowers to a
// composite omp.parallel wrapping the composite omp.distribute + omp.wsloop
// stack around a single shared omp.loop_nest. Each loop wrapper (plus the
// enclosing omp.parallel) carries the omp.composite attribute required by
// the verifier. Unlike the standalone `distribute`/`for` case, the shared loop
// counter is privatized on the omp.parallel itself (matching Flang's
// lowering of the same construct), not on omp.distribute/omp.wsloop, which
// stay bare. Here the composite is closely nested in a standalone `teams`
// region; the `teams` directive is not the innermost directive of its own
// queue (`distribute parallel for`'s decomposition still follows it), so the
// omp.teams carries omp.combined just like the plain `teams`+`distribute`
// nesting.
void distribute_parallel_for() {
  // CIR: cir.func{{.*}}@distribute_parallel_for(
#pragma omp teams
  {
#pragma omp distribute parallel for
    for (int i = 0; i < 10; i++) {
      during(i);
    }
  }

  // CIR: omp.teams {
  // The real counter's "mold" alloca must dominate omp.parallel (it becomes
  // one of its operands), so it is emitted directly inside omp.teams, before
  // omp.parallel.
  // CIR-NEXT: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CIR-NEXT: omp.parallel private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // The synthetic normalized counter is written on every iteration, so,
  // unlike the mold above, it must live inside omp.parallel's region to
  // become per-thread once the region is outlined.
  // CIR-NEXT: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>

  // The loop only runs if its precondition `0 < 10` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
  // tripCount = (10 - (0 - 1 + 1)) / 1 = 10
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
  // CIR-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
  // CIR-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32

  // CIR-NEXT: omp.distribute {
  // CIR-NEXT: omp.wsloop {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
  // CIR-NEXT: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR-NEXT: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.load align(4) %[[I_PRIV]]
  // CIR: cir.call @{{.*}}during
  // CIR: omp.yield
  // CIR: }
  // CIR: } {omp.composite}
  // CIR: } {omp.composite}
  // CIR: omp.terminator
  // CIR: } {omp.composite}
  // CIR: omp.terminator
  // CIR: } {omp.combined}
}

// The combined `teams distribute parallel for` directive nests the composite
// `distribute parallel for` stack inside an omp.teams. The `teams` directive
// is not the innermost directive, so it carries the omp.combined attribute,
// while the loop wrappers and their enclosing omp.parallel are marked
// omp.composite.
void teams_distribute_parallel_for() {
  // CIR: cir.func{{.*}}@teams_distribute_parallel_for(
#pragma omp teams distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }

  // CIR: omp.teams {
  // CIR-NEXT: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CIR-NEXT: omp.parallel private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>

  // The loop only runs if its precondition `0 < 10` holds.
  // CIR-NEXT: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
  // tripCount = (10 - (0 - 1 + 1)) / 1 = 10
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
  // CIR-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
  // CIR-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32

  // CIR-NEXT: omp.distribute {
  // CIR-NEXT: omp.wsloop {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
  // CIR-NEXT: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR-NEXT: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.load align(4) %[[I_PRIV]]
  // CIR: cir.call @{{.*}}during
  // CIR: omp.yield
  // CIR: }
  // CIR: } {omp.composite}
  // CIR: } {omp.composite}
  // CIR: omp.terminator
  // CIR: } {omp.composite}
  // CIR: omp.terminator
  // CIR: } {omp.combined}
}

// Host wrapper: __kmpc_fork_teams is identical on both paths.
//
// The CIR path's teams-outlined function just forks the parallel region
// directly via __kmpc_fork_call, since CIR's composite lowering privatizes
// the shared counter on the omp.parallel itself rather than chunking inside
// the teams-outlined function the way OGCG does. The forked function then
// schedules both the distribute and the for chunking in a single
// __kmpc_dist_for_static_init_4u call (schedule kind 34, "distribute +
// static for"), ending with a barrier (the composite's implicit `nowait`
// doesn't apply across the whole construct the way plain `distribute`'s
// does).
//
// The OGCG path instead schedules the distribute chunking (schedule kind
// 92) in the teams-outlined function, forking the parallel region once per
// *distribute chunk* via __kmpc_fork_call; the forked function then
// schedules its own static-for chunking (schedule kind 34) over just that
// chunk, with no barrier at the end.
// LLVM-LABEL: define {{.*}} void @distribute_parallel_for(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[TEAMS_OUTLINED:[^)]+]])

// LLVM:       define internal void @[[TEAMS_OUTLINED]](
// LLVM-CIR:     call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @[[PAR_OUTLINED:[^)]+]])

// LLVM-CIR:       define internal void @[[PAR_OUTLINED]](
// LLVM-CIR:         store i32 0, ptr %[[LB:.*]], align 4
// LLVM-CIR-NEXT:    store i32 9, ptr %[[UB:.*]], align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %[[STRIDE:.*]], align 4
// LLVM-CIR-NEXT:    %[[TID:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_dist_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 34,
// LLVM-CIR-SAME:      ptr %{{[^,]+}}, ptr %[[LB]], ptr %[[UB]], ptr %{{[^,]+}}, ptr %[[STRIDE]],
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %[[IV:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT:[^,]+]], %{{.*}} ]
// LLVM-CIR:         %[[LOOP_CMP:.*]] = icmp ult i32 %[[IV]], %{{.*}}
// LLVM-CIR-NEXT:    br i1 %[[LOOP_CMP]], label %[[LOOP_BODY:[^,]+]], label %[[LOOP_EXIT:[^ ,]+]]
// LLVM-CIR:       {{^}}[[LOOP_EXIT]]:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-CIR:         call void @__kmpc_barrier(
// LLVM-CIR:       {{^}}[[LOOP_BODY]]:
// LLVM-CIR:         call void @during(i32
// LLVM-CIR:         %[[NEXT]] = add nuw i32 %[[IV]], 1

// LLVM-OGCG:         store i32 0, ptr %[[COMB_LB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %[[COMB_UB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %[[DIST_STRIDE:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[DIST_IS_LAST:.*]], align 4
// LLVM-OGCG:         %[[DIST_TID:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[DIST_TID]], i32 92,
// LLVM-OGCG-SAME:      ptr %[[DIST_IS_LAST]], ptr %[[COMB_LB]], ptr %[[COMB_UB]], ptr %[[DIST_STRIDE]],
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[DIST_CMP:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[DIST_CMP]], label %[[DIST_BODY:[^,]+]], label %{{.*}}
// LLVM-OGCG:       {{^}}[[DIST_BODY]]:
// LLVM-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @[[PAR_OUTLINED:[^,]+]],
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[DIST_TID]])
// LLVM-OGCG-NEXT:    ret void

// LLVM-OGCG:       define internal void @[[PAR_OUTLINED]](
// LLVM-OGCG:         store i32 0, ptr %[[OMP_LB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB:.*]], align 4
// LLVM-OGCG:         store i32 1, ptr %[[OMP_STRIDE:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST:.*]], align 4
// LLVM-OGCG:         %[[FOR_TID:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[FOR_TID]], i32 34,
// LLVM-OGCG-SAME:      ptr %[[OMP_IS_LAST]], ptr %[[OMP_LB]], ptr %[[OMP_UB]], ptr %[[OMP_STRIDE]],
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[FOR_CMP:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[FOR_CMP]], label %[[FOR_BODY:[^,]+]], label %{{.*}}
// LLVM-OGCG:       {{^}}[[FOR_BODY]]:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[FOR_TID]])
// LLVM-OGCG-NEXT:    ret void

// `teams_distribute_parallel_for` lowers to the exact same shape as
// `distribute_parallel_for` above; we check the full CFG again here instead
// of just the call sequence, so a future change that alters this shape for
// only one of the two spellings still gets caught.
// LLVM-LABEL: define {{.*}} void @teams_distribute_parallel_for(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[TEAMS_OUTLINED2:[^)]+]])

// LLVM:       define internal void @[[TEAMS_OUTLINED2]](
// LLVM-CIR:     call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @[[PAR_OUTLINED2:[^)]+]])

// LLVM-CIR:       define internal void @[[PAR_OUTLINED2]](
// LLVM-CIR:         store i32 0, ptr %[[LB2:.*]], align 4
// LLVM-CIR-NEXT:    store i32 9, ptr %[[UB2:.*]], align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %[[STRIDE2:.*]], align 4
// LLVM-CIR-NEXT:    %[[TID2:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_dist_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 34,
// LLVM-CIR-SAME:      ptr %{{[^,]+}}, ptr %[[LB2]], ptr %[[UB2]], ptr %{{[^,]+}}, ptr %[[STRIDE2]],
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %[[IV2:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT2:[^,]+]], %{{.*}} ]
// LLVM-CIR:         %[[LOOP_CMP2:.*]] = icmp ult i32 %[[IV2]], %{{.*}}
// LLVM-CIR-NEXT:    br i1 %[[LOOP_CMP2]], label %[[LOOP_BODY2:[^,]+]], label %[[LOOP_EXIT2:[^ ,]+]]
// LLVM-CIR:       {{^}}[[LOOP_EXIT2]]:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-CIR:         call void @__kmpc_barrier(
// LLVM-CIR:       {{^}}[[LOOP_BODY2]]:
// LLVM-CIR:         call void @during(i32
// LLVM-CIR:         %[[NEXT2]] = add nuw i32 %[[IV2]], 1

// LLVM-OGCG:         store i32 0, ptr %[[COMB_LB2:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %[[COMB_UB2:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %[[DIST_STRIDE2:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[DIST_IS_LAST2:.*]], align 4
// LLVM-OGCG:         %[[DIST_TID2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[DIST_TID2]], i32 92,
// LLVM-OGCG-SAME:      ptr %[[DIST_IS_LAST2]], ptr %[[COMB_LB2]], ptr %[[COMB_UB2]], ptr %[[DIST_STRIDE2]],
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[DIST_CMP2:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[DIST_CMP2]], label %[[DIST_BODY2:[^,]+]], label %{{.*}}
// LLVM-OGCG:       {{^}}[[DIST_BODY2]]:
// LLVM-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @[[PAR_OUTLINED2:[^,]+]],
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[DIST_TID2]])
// LLVM-OGCG-NEXT:    ret void

// LLVM-OGCG:       define internal void @[[PAR_OUTLINED2]](
// LLVM-OGCG:         store i32 0, ptr %[[OMP_LB2:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB2:.*]], align 4
// LLVM-OGCG:         store i32 1, ptr %[[OMP_STRIDE2:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST2:.*]], align 4
// LLVM-OGCG:         %[[FOR_TID2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[FOR_TID2]], i32 34,
// LLVM-OGCG-SAME:      ptr %[[OMP_IS_LAST2]], ptr %[[OMP_LB2]], ptr %[[OMP_UB2]], ptr %[[OMP_STRIDE2]],
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[FOR_CMP2:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[FOR_CMP2]], label %[[FOR_BODY2:[^,]+]], label %{{.*}}
// LLVM-OGCG:       {{^}}[[FOR_BODY2]]:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[FOR_TID2]])
// LLVM-OGCG-NEXT:    ret void
