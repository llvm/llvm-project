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
  // CIR: cir.func{{.*}}@distribute_parallel_for
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
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CIR: omp.parallel private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {

  // The synthetic normalized counter is written on every iteration, so,
  // unlike the mold above, it must live inside omp.parallel's region to
  // become per-thread once the region is outlined.
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[C0:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
  // CIR: %[[C1:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
  // CIR: %[[C10:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32

  // CIR: omp.distribute {
  // CIR-NEXT: omp.wsloop {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {
  // CIR: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
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
  // CIR: cir.func{{.*}}@teams_distribute_parallel_for
#pragma omp teams distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }

  // CIR: omp.teams {
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CIR: omp.parallel private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR: omp.distribute {
  // CIR-NEXT: omp.wsloop {
  // CIR-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CIR: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
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

// Host wrapper: __kmpc_fork_teams is identical on both paths (only the
// pointer to the teams-outlined function differs, absorbed by the
// wildcard below).
//
// LLVM-LABEL: define {{.*}} void @distribute_parallel_for(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @distribute_parallel_for{{.*}})

// The CIR path's teams-outlined function just forks the parallel region
// directly via __kmpc_fork_call, since CIR's composite lowering privatizes
// the shared counter on the omp.parallel itself rather than chunking inside
// the teams-outlined function the way OGCG does (compare the OGCG blocks
// further down, which do their own distribute chunking here instead).
// LLVM-CIR-LABEL: define internal void @distribute_parallel_for..omp_par{{.*}}(
// LLVM-CIR:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @distribute_parallel_for..omp_par{{.*}})

// CIR schedules both the distribute and the for chunking in a single
// __kmpc_dist_for_static_init_4u call (schedule kind 34, "distribute +
// static for"), ending with a barrier (the composite's implicit `nowait`
// doesn't apply across the whole construct the way plain `distribute`'s
// does). CIR's exit block happens to print before its body block even
// though body always runs first, so we check the actual branch targets
// below instead of relying on the printed order (same quirk as
// teams-distribute.c).
// LLVM-CIR-LABEL: define internal void @distribute_parallel_for..omp_par{{.*}}(
// LLVM-CIR:         %[[TID:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_dist_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 34,
// LLVM-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.distupperbound, ptr %p.stride,
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-CIR:       omp_loop.exit:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-CIR:         call void @__kmpc_barrier(
// LLVM-CIR:       omp_loop.body:
// LLVM-CIR:         call void @during(i32

// The OGCG path instead schedules the distribute chunking (schedule kind
// 92) in the teams-outlined function, forking the parallel region once per
// *distribute chunk* via __kmpc_fork_call; the forked function then
// schedules its own static-for chunking (schedule kind 34) over just that
// chunk, with no barrier at the end.
// LLVM-OGCG-LABEL: define internal void @distribute_parallel_for.omp_outlined(
// LLVM-OGCG:         store i32 0, ptr %.omp.comb.lb, align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %.omp.comb.ub, align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// LLVM-OGCG:         %[[TID3:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 92,
// LLVM-OGCG-SAME:      ptr %.omp.is_last, ptr %.omp.comb.lb, ptr %.omp.comb.ub, ptr %.omp.stride,
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:       omp.inner.for.body:
// LLVM-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @distribute_parallel_for.omp_outlined.omp_outlined,
// LLVM-OGCG:       omp.loop.exit:
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-OGCG-NEXT:    ret void

// LLVM-OGCG-LABEL: define internal void @distribute_parallel_for.omp_outlined.omp_outlined(
// LLVM-OGCG:         %[[TID4:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID4]], i32 34,
// LLVM-OGCG:       omp.inner.for.body:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:       omp.loop.exit:
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID4]])
// LLVM-OGCG-NEXT:    ret void

// `teams_distribute_parallel_for` lowers to the exact same shape as
// `distribute_parallel_for` above; we check the full CFG again here instead
// of just the call sequence, so a future change that alters this shape for
// only one of the two spellings still gets caught.
// LLVM-LABEL: define {{.*}} void @teams_distribute_parallel_for(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @teams_distribute_parallel_for{{.*}})

// LLVM-CIR-LABEL: define internal void @teams_distribute_parallel_for..omp_par{{.*}}(
// LLVM-CIR:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @teams_distribute_parallel_for..omp_par{{.*}})

// LLVM-CIR-LABEL: define internal void @teams_distribute_parallel_for..omp_par{{.*}}(
// LLVM-CIR:         %[[TID2:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_dist_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 34,
// LLVM-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.distupperbound, ptr %p.stride,
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-CIR:       omp_loop.exit:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-CIR:         call void @__kmpc_barrier(
// LLVM-CIR:       omp_loop.body:
// LLVM-CIR:         call void @during(i32

// `teams_distribute_parallel_for` lowers to the exact same shape as
// `distribute_parallel_for` above on the OGCG path too.
// LLVM-OGCG-LABEL: define internal void @teams_distribute_parallel_for.omp_outlined(
// LLVM-OGCG:         %[[TID5:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID5]], i32 92,
// LLVM-OGCG:       omp.inner.for.body:
// LLVM-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @teams_distribute_parallel_for.omp_outlined.omp_outlined,
// LLVM-OGCG:       omp.loop.exit:
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID5]])
// LLVM-OGCG-NEXT:    ret void

// LLVM-OGCG-LABEL: define internal void @teams_distribute_parallel_for.omp_outlined.omp_outlined(
// LLVM-OGCG:         %[[TID6:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID6]], i32 34,
// LLVM-OGCG:       omp.inner.for.body:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:       omp.loop.exit:
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID6]])
// LLVM-OGCG-NEXT:    ret void
