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

  // CIR-NEXT: omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
  // CIR-NEXT: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CIR-NEXT: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
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
  // CIR-NEXT: %[[IV_ALLOCA2:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
  // CIR-NEXT: %[[I_ALLOCA2:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>

  // The loop only runs if its precondition `0 < 10` holds.
  // CIR-NEXT: %[[PRE_L2:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[PRE_R2:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[PRECOND2:.*]] = cir.cmp lt %[[PRE_L2]], %[[PRE_R2]] : !s32i
  // tripCount = (10 - (0 - 1 + 1)) / 1 = 10
  // CIR-NEXT: %[[HI2:.*]] = cir.const #cir.int<10> : !s32i
  // CIR-NEXT: %[[LO2:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[STEP_A2:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[LO_MINUS_STEP2:.*]] = cir.sub nsw %[[LO2]], %[[STEP_A2]] : !s32i
  // CIR-NEXT: %[[ONE_A2:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ADJ_LO2:.*]] = cir.add nsw %[[LO_MINUS_STEP2]], %[[ONE_A2]] : !s32i
  // CIR-NEXT: %[[SPAN2:.*]] = cir.sub nsw %[[HI2]], %[[ADJ_LO2]] : !s32i
  // CIR-NEXT: %[[STEP_B2:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_CIR2:.*]] = cir.div %[[SPAN2]], %[[STEP_B2]] : !s32i
  // CIR-NEXT: %[[TC_ZERO2:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[TRIPCOUNT_SEL2:.*]] = cir.select if %[[PRECOND2]] then %[[TRIPCOUNT_CIR2]] else %[[TC_ZERO2]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR-NEXT: %[[ZERO_CIR2:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[ZERO2:.*]] = cir.builtin_int_cast %[[ZERO_CIR2]] : !s32i -> i32
  // CIR-NEXT: %[[ONE_CIR2:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT: %[[ONE2:.*]] = cir.builtin_int_cast %[[ONE_CIR2]] : !s32i -> i32
  // CIR-NEXT: %[[TRIPCOUNT2:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL2]] : !s32i -> i32

  // CIR-NEXT: omp.distribute private(@{{.*}} %[[I_ALLOCA2]] -> %[[I_PRIV2:.*]] : !cir.ptr<!s32i>) {
  // CIR-NEXT: omp.loop_nest (%[[IV2:.*]]) : i32 = (%[[ZERO2]]) to (%[[TRIPCOUNT2]]) step (%[[ONE2]]) {
  // CIR-NEXT: %[[IV_CIR2:.*]] = cir.builtin_int_cast %[[IV2]] : i32 -> !s32i
  // CIR-NEXT: cir.store align(4) %[[IV_CIR2]], %[[IV_ALLOCA2]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) %{{.*}}, %[[I_PRIV2]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.load align(4) %[[I_PRIV2]]
  // CIR: cir.call @{{.*}}during
  // CIR: omp.yield
  // CIR: }
  // CIR: }
  // CIR: omp.terminator
  // CIR: } {omp.combined}
}

// Host wrapper: __kmpc_fork_teams is identical on both paths. The
// distribute loop inside the teams-outlined function uses schedule 92
// (static distribute) with a different runtime API and control-flow shape
// on each path.
//
// LLVM-LABEL: define {{.*}} void @explicit_teams_distribute(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[OUTLINED:[^)]+]])

// LLVM-CIR:       define internal void @[[OUTLINED]](
// LLVM-CIR:         store i32 0, ptr %[[LB:.*]], align 4
// LLVM-CIR-NEXT:    store i32 9, ptr %[[UB:.*]], align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %[[STRIDE:.*]], align 4
// LLVM-CIR-NEXT:    %[[TID:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 92,
// LLVM-CIR-SAME:      ptr %{{[^,]+}}, ptr %[[LB]], ptr %[[UB]], ptr %[[STRIDE]],
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %[[IV:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT:[^,]+]], %{{.*}} ]
// LLVM-CIR:         %[[LOOP_CMP:.*]] = icmp ult i32 %[[IV]], %{{.*}}
// LLVM-CIR-NEXT:    br i1 %[[LOOP_CMP]], label %[[LOOP_BODY:[^,]+]], label %[[LOOP_EXIT:[^ ,]+]]
// LLVM-CIR:       {{^}}[[LOOP_EXIT]]:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-CIR:       {{^}}[[LOOP_BODY]]:
// LLVM-CIR:         call void @during(i32
// LLVM-CIR:         %[[NEXT]] = add nuw i32 %[[IV]], 1

// LLVM-OGCG:       define internal void @[[OUTLINED]](
// LLVM-OGCG:         store i32 0, ptr %[[OMP_LB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST:.*]], align 4
// LLVM-OGCG:         %[[TID_OGCG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID_OGCG]], i32 92,
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
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID_OGCG]])
// LLVM-OGCG-NEXT:    ret void

// `teams_distribute`
// LLVM-LABEL: define {{.*}} void @teams_distribute(
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @[[OUTLINED2:[^)]+]])

// LLVM-CIR:       define internal void @[[OUTLINED2]](
// LLVM-CIR:         store i32 0, ptr %[[LB2:.*]], align 4
// LLVM-CIR-NEXT:    store i32 9, ptr %[[UB2:.*]], align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %[[STRIDE2:.*]], align 4
// LLVM-CIR-NEXT:    %[[TID2:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 92,
// LLVM-CIR-SAME:      ptr %{{[^,]+}}, ptr %[[LB2]], ptr %[[UB2]], ptr %[[STRIDE2]],
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %[[IV2:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT2:[^,]+]], %{{.*}} ]
// LLVM-CIR:         %[[LOOP_CMP2:.*]] = icmp ult i32 %[[IV2]], %{{.*}}
// LLVM-CIR-NEXT:    br i1 %[[LOOP_CMP2]], label %[[LOOP_BODY2:[^,]+]], label %[[LOOP_EXIT2:[^ ,]+]]
// LLVM-CIR:       {{^}}[[LOOP_EXIT2]]:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-CIR:       {{^}}[[LOOP_BODY2]]:
// LLVM-CIR:         call void @during(i32
// LLVM-CIR:         %[[NEXT2]] = add nuw i32 %[[IV2]], 1

// LLVM-OGCG:       define internal void @[[OUTLINED2]](
// LLVM-OGCG:         store i32 0, ptr %[[OMP_LB2:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB2:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE2:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST2:.*]], align 4
// LLVM-OGCG:         %[[TID_OGCG2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID_OGCG2]], i32 92,
// LLVM-OGCG-SAME:      ptr %[[OMP_IS_LAST2]], ptr %[[OMP_LB2]], ptr %[[OMP_UB2]], ptr %[[OMP_STRIDE2]],
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[UB_VAL2:.*]] = load i32, ptr %[[OMP_UB2]], align 4
// LLVM-OGCG-NEXT:    %[[CMP2:.*]] = icmp sgt i32 %[[UB_VAL2]], 9
// LLVM-OGCG-NEXT:    br i1 %[[CMP2]], label %[[COND_TRUE2:[^,]+]], label %[[COND_FALSE2:[^ ,]+]]
// LLVM-OGCG:         %[[COND2:.*]] = phi i32 [ 9, %[[COND_TRUE2]] ], [ %{{.*}}, %[[COND_FALSE2]] ]
// LLVM-OGCG-NEXT:    store i32 %[[COND2]], ptr %[[OMP_UB2]], align 4
// LLVM-OGCG:         %[[CMP12:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP12]], label %[[FOR_BODY2:[^,]+]], label %{{.*}}
// LLVM-OGCG:       {{^}}[[FOR_BODY2]]:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID_OGCG2]])
// LLVM-OGCG-NEXT:    ret void
