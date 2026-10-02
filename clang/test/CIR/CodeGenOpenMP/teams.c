// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR

// Lowering to LLVM IR: mostly identical between CIR and OGCG; they just name
// the outlined functions differently.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-OGCG

void use(int);

// A standalone `teams` directive lowers to an omp.teams op whose region holds
// the associated structured block.
void host_teams(int x) {
  // CIR: cir.func{{.*}}@host_teams
  // CIR: omp.teams {
  // CIR: cir.call @use
  // CIR: omp.terminator
  // CIR: }
#pragma omp teams
  {
    use(x);
  }
}

// LLVM-LABEL: define {{.*}} void @host_teams(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @host_teams{{.*}}, ptr {{.*}})

// LLVM-CIR-LABEL: define internal void @host_teams..omp_par(
// LLVM-OGCG-LABEL: define internal void @host_teams.omp_outlined(
// LLVM:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

void teams_nested_parallel_for(void) {
  // CIR: cir.func{{.*}}@teams_nested_parallel_for
  // CIR: omp.teams {
  // CIR: omp.parallel {
  // CIR: omp.wsloop private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest
  // CIR: cir.call @use
  // CIR: omp.yield
  // CIR: }
  // CIR: omp.terminator
  // CIR: } {omp.combined}
  // CIR: omp.terminator
  // CIR: } {omp.combined}
#pragma omp teams
  {
#pragma omp parallel for
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// The nested `parallel for` forks its own threads via __kmpc_fork_call from
// inside the teams-outlined function, so the wsloop's init/fini/barrier
// live in a second, further-outlined function.
// LLVM-LABEL: define {{.*}} void @teams_nested_parallel_for(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @teams_nested_parallel_for{{.*}})

// LLVM-CIR-LABEL: define internal void @teams_nested_parallel_for..omp_par.1(
// LLVM-OGCG-LABEL: define internal void @teams_nested_parallel_for.omp_outlined(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @teams_nested_parallel_for{{.*}})

// OGCG skips the trailing barrier after the worksharing loop (the
// implicit join at the end of the parallel region already synchronizes);
// CIR doesn't yet apply that optimization, so it emits one either way.
// LLVM-CIR-LABEL: define internal void @teams_nested_parallel_for..omp_par(
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
// LLVM-CIR:         call void @use(i32

// LLVM-OGCG-LABEL: define internal void @teams_nested_parallel_for.omp_outlined.omp_outlined(
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
// LLVM-OGCG:         call void @use(i32
// LLVM-OGCG:       omp.loop.exit:
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-OGCG-NEXT:    ret void
