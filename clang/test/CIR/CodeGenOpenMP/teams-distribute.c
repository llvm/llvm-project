// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR

// Lowering to LLVM IR: mostly identical between CIR and OGCG; they just name
// the outlined functions differently.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-OGCG

void during(int);

// Explicit `teams` + `distribute` nesting lowers the same as the combined
// `teams distribute` spelling below, so both get omp.combined.
void explicit_teams_distribute() {
  // CIR: cir.func{{.*}}@explicit_teams_distribute
#pragma omp teams
  {
#pragma omp distribute
    for (int i = 0; i < 10; i++) {
      during(i);
    }
  }

  // CIR: omp.teams {
  // CIR: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
  // CIR: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
  // CIR: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
  // CIR: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
  // CIR: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32

  // CIR: omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[STEP]]) {
  // CIR: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.load align(4) %[[I_PRIV]]
  // CIR: cir.call @{{.*}}during
  // CIR: omp.yield
  // CIR: }
  // CIR: }
  // CIR: omp.terminator
  // CIR: } {omp.combined}
}

// Combined `teams distribute` lowers the same as the explicit nesting
// above, with omp.combined on the omp.teams.
void teams_distribute() {
  // CIR: cir.func{{.*}}@teams_distribute
#pragma omp teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }

  // CIR: omp.teams {
  // CIR: omp.distribute private(@{{.*}} %{{.*}} -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CIR: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.call @{{.*}}during
  // CIR: omp.yield
  // CIR: }
  // CIR: }
  // CIR: omp.terminator
  // CIR: } {omp.combined}
}

// Host wrapper: __kmpc_fork_teams is identical on both paths.
//
// LLVM-LABEL: define {{.*}} void @explicit_teams_distribute(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @explicit_teams_distribute{{.*}})

// LLVM-CIR-LABEL: define internal void @explicit_teams_distribute..omp_par(
// LLVM-CIR:         %[[TID:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 92,
// LLVM-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-CIR:       omp_loop.exit:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-CIR:       omp_loop.body:
// LLVM-CIR:         call void @during(i32

// LLVM-OGCG-LABEL: define internal void @explicit_teams_distribute.omp_outlined(
// LLVM-OGCG:         store i32 0, ptr %.omp.lb, align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %.omp.ub, align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// LLVM-OGCG:         %[[TID2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 92,
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
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-OGCG-NEXT:    ret void

// `teams_distribute`
// LLVM-LABEL: define {{.*}} void @teams_distribute(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @teams_distribute{{.*}})

// LLVM-CIR-LABEL: define internal void @teams_distribute..omp_par(
// LLVM-CIR:         %[[TID3:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 92,
// LLVM-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-CIR:       omp_loop.exit:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-CIR:       omp_loop.body:
// LLVM-CIR:         call void @during(i32

// LLVM-OGCG-LABEL: define internal void @teams_distribute.omp_outlined(
// LLVM-OGCG:         store i32 0, ptr %.omp.lb, align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %.omp.ub, align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// LLVM-OGCG:         %[[TID4:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID4]], i32 92,
// LLVM-OGCG-SAME:      ptr %.omp.is_last, ptr %.omp.lb, ptr %.omp.ub, ptr %.omp.stride,
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[UB2:.*]] = load i32, ptr %.omp.ub, align 4
// LLVM-OGCG-NEXT:    %[[CMP2:.*]] = icmp sgt i32 %[[UB2]], 9
// LLVM-OGCG-NEXT:    br i1 %[[CMP2]], label %cond.true, label %cond.false
// LLVM-OGCG:       cond.end:
// LLVM-OGCG:         %[[COND2:.*]] = phi i32 [ 9, %cond.true ], [ %{{.*}}, %cond.false ]
// LLVM-OGCG-NEXT:    store i32 %[[COND2]], ptr %.omp.ub, align 4
// LLVM-OGCG:       omp.inner.for.cond:
// LLVM-OGCG:         %[[CMP3:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP3]], label %omp.inner.for.body, label %omp.inner.for.end
// LLVM-OGCG:       omp.inner.for.body:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:       omp.loop.exit:
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID4]])
// LLVM-OGCG-NEXT:    ret void
