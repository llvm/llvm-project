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
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @[[OUTLINED:[^,]+]], ptr {{.*}})

// LLVM:       define internal void @[[OUTLINED]](
// LLVM:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

void teams_nested_parallel_for(void) {
  // CIR: cir.func{{.*}}@teams_nested_parallel_for
  // CIR: omp.teams {
  // CIR: omp.parallel {
  // CIR-NEXT: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR-NEXT: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>

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

  // CIR-NEXT: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
  // CIR-NEXT: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR-NEXT: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
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
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[TEAMS_OUTLINED:[^)]+]])

// LLVM:       define internal void @[[TEAMS_OUTLINED]](
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @[[PAR_OUTLINED:[^)]+]])

// OGCG skips the trailing barrier after the worksharing loop (the
// implicit join at the end of the parallel region already synchronizes);
// CIR doesn't yet apply that optimization, so it emits one either way.
// LLVM-CIR:       define internal void @[[PAR_OUTLINED]](
// LLVM-CIR:         %[[LASTITER:.*]] = alloca i32, align 4
// LLVM-CIR-NEXT:    %[[LB:.*]] = alloca i32, align 4
// LLVM-CIR-NEXT:    %[[UB:.*]] = alloca i32, align 4
// LLVM-CIR-NEXT:    %[[STRIDE:.*]] = alloca i32, align 4
// LLVM-CIR:         store i32 0, ptr %[[LB]], align 4
// LLVM-CIR-NEXT:    store i32 9, ptr %[[UB]], align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %[[STRIDE]], align 4
// LLVM-CIR-NEXT:    %[[TID:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
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
// LLVM-CIR:         call void @use(i32
// LLVM-CIR:         %[[NEXT]] = add nuw i32 %[[IV]], 1

// LLVM-OGCG:       define internal void @[[PAR_OUTLINED]](
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
// LLVM-OGCG:         call void @use(i32
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-OGCG-NEXT:    ret void
