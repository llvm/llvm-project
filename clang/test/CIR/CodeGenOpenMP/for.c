// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM,LLVM-OGCG

void before(int);
void during(int);
void after(int);

void emit_simple_for() {
  int j = 5;
  before(j);
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
      during(j);
    }
  }
  after(j);
}

// The before()/after() calls around the parallel region are identical on
// both paths.
//
// LLVM-LABEL: define{{.*}} void @emit_simple_for()
// LLVM:         %[[J_SLOT:.*]] = alloca i32, align 4
// LLVM:         store i32 5, ptr %[[J_SLOT]], align 4
// LLVM:         %[[J_VAL:.*]] = load i32, ptr %[[J_SLOT]], align 4
// LLVM-NEXT:    call void @before(i32 noundef %[[J_VAL]])

// CIR packs the captured variable(s) into a struct and passes a single
// pointer to that struct as the __kmpc_fork_call vararg; OGCG instead
// passes a direct pointer to each captured variable as its own vararg.
// The forked function's signature and body differ accordingly (GEP+load
// from the struct vs a plain by-reference parameter).
// LLVM-CIR:         %[[GEP0:.*]] = getelementptr { ptr }, ptr %[[STRUCT_ARG:.*]], i32 0, i32 0
// LLVM-CIR-NEXT:    store ptr %[[J_SLOT]], ptr %[[GEP0]], align 8
// LLVM-CIR:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @[[SIMPLE_OUTLINED:[^,]+]], ptr %[[STRUCT_ARG]])

// LLVM-OGCG:        call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @[[SIMPLE_OUTLINED:[^,]+]], ptr %[[J_SLOT]])

// LLVM:         %[[J_VAL2:.*]] = load i32, ptr %[[J_SLOT]], align 4
// LLVM-NEXT:    call void @after(i32 noundef %[[J_VAL2]])
// LLVM-NEXT:    ret void

// The worksharing loop: CIR uses the unsigned-trip-count
// __kmpc_for_static_init_4u/fini pair and synthesizes a canonical
// omp_loop.header/cond/body/inc/exit skeleton; OGCG uses the signed
// __kmpc_for_static_init_4/fini pair with a classic cond.true/false/end
// clamp and an omp.inner.for.cond/body/inc skeleton.
//
// LLVM-CIR:       define internal void @[[SIMPLE_OUTLINED]](
// LLVM-CIR-SAME:    ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr noalias nofreeobj %[[STRUCT:.*]])
// LLVM-CIR:         %[[GEP_J:.*]] = getelementptr { ptr }, ptr %[[STRUCT]], i32 0, i32 0
// LLVM-CIR-NEXT:    %[[J_PTR:.*]] = load ptr, ptr %[[GEP_J]], align 8
// LLVM-CIR-NEXT:    %[[LASTITER:.*]] = alloca i32, align 4
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
// LLVM-CIR:         %[[J_VAL3:.*]] = load i32, ptr %[[J_PTR]], align 4
// LLVM-CIR-NEXT:    call void @during(i32 noundef %[[J_VAL3]])
// LLVM-CIR:         %[[NEXT]] = add nuw i32 %[[IV]], 1
// LLVM-CIR:         ret void

// LLVM-OGCG:       define internal void @[[SIMPLE_OUTLINED]](
// LLVM-OGCG-SAME:    ptr noalias noundef %[[GTID:[^,]+]], ptr noalias noundef %{{[^,]+}},
// LLVM-OGCG-SAME:    ptr nofreeobj noundef nonnull align 4 dereferenceable(4) %[[J:[^)]+]])
// LLVM-OGCG:         %[[GTID_ADDR:.*]] = alloca ptr, align 8
// LLVM-OGCG-NEXT:    %{{.*}} = alloca ptr, align 8
// LLVM-OGCG-NEXT:    %[[J_ADDR:.*]] = alloca ptr, align 8
// LLVM-OGCG:         store ptr %[[GTID]], ptr %[[GTID_ADDR]], align 8
// LLVM-OGCG:         store ptr %[[J]], ptr %[[J_ADDR]], align 8
// LLVM-OGCG:         %[[J_LOADED:.*]] = load ptr, ptr %[[J_ADDR]], align 8
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[OMP_LB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE:.*]], align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST:.*]], align 4
// LLVM-OGCG:         %[[GTID_VAL_PTR:.*]] = load ptr, ptr %[[GTID_ADDR]], align 8
// LLVM-OGCG-NEXT:    %[[TID2:.*]] = load i32, ptr %[[GTID_VAL_PTR]], align 4
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
// LLVM-OGCG:         %[[J_VAL3:.*]] = load i32, ptr %[[J_LOADED]], align 4
// LLVM-OGCG-NEXT:    call void @during(i32 noundef %[[J_VAL3]])
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-OGCG-NEXT:    call void @__kmpc_barrier(ptr {{.*}}, i32 %[[TID2]])
// LLVM-OGCG-NEXT:    ret void

void emit_for_with_vars() {
  int j = 5;
  before(j);
#pragma omp parallel
  {
    int lb = 1;
    long ub = 10;
    short step = 1;
#pragma omp for
    for (int i = 0; i < ub; i = i + step) {
      during(j);
    }
  }
  after(j);
}

// LLVM-LABEL: define{{.*}} void @emit_for_with_vars()

// Same CIR-vs-OGCG capture-ABI split as emit_simple_for above.
// LLVM-CIR:      %[[STRUCT_ARG2:.*]] = alloca { ptr }, align 8

// LLVM:         call void @before(i32

// LLVM-CIR:      call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @[[VARS_OUTLINED:[^,]+]], ptr %[[STRUCT_ARG2]])

// LLVM-OGCG:     call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @[[VARS_OUTLINED:[^,]+]], ptr {{.*}})

// LLVM:         call void @after(i32
// LLVM-NEXT:    ret void

// Loop bounds aren't known until runtime (`ub` and `step` are variables, not
// constants), so both paths have to compute the number of iterations on the
// fly. We check the loop setup, during() call (one per iteration), and the
// matching init/fini/barrier call betwen CIR and OGCG.
//
// LLVM-CIR:       define internal void @[[VARS_OUTLINED]](
// LLVM-CIR:         store i32 1, ptr %[[LB:.*]], align 4
// LLVM-CIR-NEXT:    store i64 10, ptr %[[UB:.*]], align 8
// LLVM-CIR-NEXT:    store i16 1, ptr %[[STEP:.*]], align 2
// LLVM-CIR:         %[[STEP_W:.*]] = sext i16 %{{.*}} to i64
// LLVM-CIR-NEXT:    %[[LO_MINUS_STEP:.*]] = sub nsw i64 0, %[[STEP_W]]
// LLVM-CIR-NEXT:    %[[LO_ADJ:.*]] = add nsw i64 %[[LO_MINUS_STEP]], 1
// LLVM-CIR-NEXT:    %[[SPAN:.*]] = sub nsw i64 %{{.*}}, %[[LO_ADJ]]
// LLVM-CIR-NEXT:    %[[STEP_V2:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM-CIR-NEXT:    %[[STEP_W2:.*]] = sext i16 %[[STEP_V2]] to i64
// LLVM-CIR-NEXT:    %[[NITERS:.*]] = sdiv i64 %[[SPAN]], %[[STEP_W2]]
// LLVM-CIR-NEXT:    %{{.*}} = trunc i64 %[[NITERS]] to i32
// LLVM-CIR:         %[[TRIPCOUNT:.*]] = select i1 %{{.*}}, i32 0, i32 %{{.*}}
// LLVM-CIR:         store i32 0, ptr %[[LB2:.*]], align 4
// LLVM-CIR-NEXT:    %[[TRIPM1:.*]] = sub i32 %[[TRIPCOUNT]], 1
// LLVM-CIR-NEXT:    store i32 %[[TRIPM1]], ptr %[[UB2:.*]], align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %[[STRIDE2:.*]], align 4
// LLVM-CIR:         %[[TID3:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 34,
// LLVM-CIR-SAME:      ptr %{{.*}}, ptr %[[LB2]], ptr %[[UB2]], ptr %[[STRIDE2]],
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %[[IV2:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT2:[^,]+]], %{{.*}} ]
// LLVM-CIR:         %[[LOOP_CMP2:.*]] = icmp ult i32 %[[IV2]], %{{.*}}
// LLVM-CIR-NEXT:    br i1 %[[LOOP_CMP2]], label %[[LOOP_BODY2:[^,]+]], label %[[LOOP_EXIT2:[^ ,]+]]
// LLVM-CIR:       {{^}}[[LOOP_EXIT2]]:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-CIR:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-CIR:       {{^}}[[LOOP_BODY2]]:
// LLVM-CIR:         call void @during(i32
// LLVM-CIR:         %[[NEXT2]] = add nuw i32 %[[IV2]], 1
// LLVM-CIR:         ret void

// LLVM-OGCG:       define internal void @[[VARS_OUTLINED]](
// LLVM-OGCG-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}},
// LLVM-OGCG-SAME:    ptr nofreeobj noundef nonnull align 4 dereferenceable(4) %{{[^)]+}})
// LLVM-OGCG:         %[[LB_VAR:.*]] = alloca i32, align 4
// LLVM-OGCG-NEXT:    %[[UB_VAR:.*]] = alloca i64, align 8
// LLVM-OGCG-NEXT:    %[[STEP_VAR:.*]] = alloca i16, align 2
// LLVM-OGCG:         store i32 1, ptr %[[LB_VAR]], align 4
// LLVM-OGCG-NEXT:    store i64 10, ptr %[[UB_VAR]], align 8
// LLVM-OGCG-NEXT:    store i16 1, ptr %[[STEP_VAR]], align 2
// LLVM-OGCG:         %[[STEP_W:.*]] = sext i16 %{{.*}} to i64
// LLVM-OGCG-NEXT:    %[[LO_MINUS_STEP:.*]] = sub nsw i64 0, %[[STEP_W]]
// LLVM-OGCG-NEXT:    %[[LO_ADJ:.*]] = add nsw i64 %[[LO_MINUS_STEP]], 1
// LLVM-OGCG-NEXT:    %[[SPAN:.*]] = sub nsw i64 %{{.*}}, %[[LO_ADJ]]
// LLVM-OGCG-NEXT:    %[[STEP_V2:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM-OGCG-NEXT:    %[[STEP_W2:.*]] = sext i16 %[[STEP_V2]] to i64
// LLVM-OGCG-NEXT:    %[[NITERS:.*]] = sdiv i64 %[[SPAN]], %[[STEP_W2]]
// LLVM-OGCG-NEXT:    %{{.*}} = trunc i64 %[[NITERS]] to i32
// LLVM-OGCG:         %[[PRECOND:.*]] = icmp slt i64 0, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[PRECOND]], label %[[PRECOND_THEN:[^,]+]], label %[[PRECOND_END:[^ ,]+]]
// LLVM-OGCG:       {{^}}[[PRECOND_THEN]]:
// LLVM-OGCG-NEXT:    store i32 0, ptr %[[OMP_LB2:.*]], align 4
// LLVM-OGCG:         store i32 1, ptr %[[OMP_STRIDE2:.*]], align 4
// LLVM-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %{{.*}}, i32 34,
// LLVM-OGCG-SAME:      ptr %{{.*}}, ptr %[[OMP_LB2]], ptr %[[OMP_UB2:[^,]+]], ptr %[[OMP_STRIDE2]],
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[CMP9:.*]] = icmp sgt i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP9]], label %[[COND_TRUE2:[^,]+]], label %[[COND_FALSE2:[^ ,]+]]
// LLVM-OGCG:         %[[COND2:.*]] = phi i32 [ %{{.*}}, %[[COND_TRUE2]] ], [ %{{.*}}, %[[COND_FALSE2]] ]
// LLVM-OGCG-NEXT:    store i32 %[[COND2]], ptr %[[OMP_UB2]], align 4
// LLVM-OGCG:         %[[CMP11:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP11]], label %[[FOR_BODY2:[^,]+]], label %{{.*}}
// LLVM-OGCG:       {{^}}[[FOR_BODY2]]:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %{{.*}})
// LLVM-OGCG:       {{^}}[[PRECOND_END]]:
// LLVM-OGCG:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-OGCG-NEXT:    ret void
