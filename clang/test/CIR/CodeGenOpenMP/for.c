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
// LLVM-CIR:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @emit_simple_for..omp_par, ptr %[[STRUCT_ARG]])

// LLVM-OGCG:        call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @emit_simple_for.omp_outlined, ptr %[[J_SLOT]])

// LLVM:         %[[J_VAL2:.*]] = load i32, ptr %[[J_SLOT]], align 4
// LLVM-NEXT:    call void @after(i32 noundef %[[J_VAL2]])
// LLVM-NEXT:    ret void

// The worksharing loop: CIR uses the unsigned-trip-count
// __kmpc_for_static_init_4u/fini pair and synthesizes a canonical
// omp_loop.header/cond/body/inc/exit skeleton; OGCG uses the signed
// __kmpc_for_static_init_4/fini pair with a classic cond.true/false/end
// clamp and an omp.inner.for.cond/body/inc skeleton.
//
// LLVM-CIR-LABEL: define internal void @emit_simple_for..omp_par(
// LLVM-CIR-SAME:    ptr noalias %tid.addr, ptr noalias %zero.addr, ptr noalias nofreeobj %[[STRUCT:.*]])
// LLVM-CIR:         %[[GEP_J:.*]] = getelementptr { ptr }, ptr %[[STRUCT]], i32 0, i32 0
// LLVM-CIR-NEXT:    %[[J_PTR:.*]] = load ptr, ptr %[[GEP_J]], align 8
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
// LLVM-CIR:         %[[J_VAL3:.*]] = load i32, ptr %[[J_PTR]], align 4
// LLVM-CIR-NEXT:    call void @during(i32 noundef %[[J_VAL3]])
// LLVM-CIR:         %omp_loop.next = add nuw i32 %omp_loop.iv, 1
// LLVM-CIR:         ret void

// LLVM-OGCG-LABEL: define internal void @emit_simple_for.omp_outlined(
// LLVM-OGCG-SAME:    ptr noalias noundef %.global_tid., ptr noalias noundef %.bound_tid.,
// LLVM-OGCG-SAME:    ptr nofreeobj noundef nonnull align 4 dereferenceable(4) %j)
// LLVM-OGCG:         %.global_tid..addr = alloca ptr, align 8
// LLVM-OGCG-NEXT:    %.bound_tid..addr = alloca ptr, align 8
// LLVM-OGCG-NEXT:    %j.addr = alloca ptr, align 8
// LLVM-OGCG:         %.omp.lb = alloca i32, align 4
// LLVM-OGCG-NEXT:    %.omp.ub = alloca i32, align 4
// LLVM-OGCG-NEXT:    %.omp.stride = alloca i32, align 4
// LLVM-OGCG-NEXT:    %.omp.is_last = alloca i32, align 4
// LLVM-OGCG:         store ptr %.global_tid., ptr %.global_tid..addr, align 8
// LLVM-OGCG:         store ptr %j, ptr %j.addr, align 8
// LLVM-OGCG:         %[[J_LOADED:.*]] = load ptr, ptr %j.addr, align 8
// LLVM-OGCG:         store i32 0, ptr %.omp.lb, align 4
// LLVM-OGCG-NEXT:    store i32 9, ptr %.omp.ub, align 4
// LLVM-OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// LLVM-OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// LLVM-OGCG:         %[[GTID_VAL_PTR:.*]] = load ptr, ptr %.global_tid..addr, align 8
// LLVM-OGCG-NEXT:    %[[TID2:.*]] = load i32, ptr %[[GTID_VAL_PTR]], align 4
// LLVM-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 34,
// LLVM-OGCG-SAME:      ptr %.omp.is_last, ptr %.omp.lb, ptr %.omp.ub, ptr %.omp.stride,
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[UB_VAL:.*]] = load i32, ptr %.omp.ub, align 4
// LLVM-OGCG-NEXT:    %[[CMP:.*]] = icmp sgt i32 %[[UB_VAL]], 9
// LLVM-OGCG-NEXT:    br i1 %[[CMP]], label %cond.true, label %cond.false
// LLVM-OGCG:       cond.end:
// LLVM-OGCG:         %[[COND:.*]] = phi i32 [ 9, %cond.true ], [ %{{.*}}, %cond.false ]
// LLVM-OGCG-NEXT:    store i32 %[[COND]], ptr %.omp.ub, align 4
// LLVM-OGCG:       omp.inner.for.cond:
// LLVM-OGCG:         %[[CMP1:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP1]], label %omp.inner.for.body, label %omp.inner.for.end
// LLVM-OGCG:       omp.inner.for.body:
// LLVM-OGCG:         %[[J_VAL3:.*]] = load i32, ptr %[[J_LOADED]], align 4
// LLVM-OGCG-NEXT:    call void @during(i32 noundef %[[J_VAL3]])
// LLVM-OGCG:       omp.loop.exit:
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

// LLVM-CIR:      call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @emit_for_with_vars..omp_par, ptr %[[STRUCT_ARG2]])

// LLVM-OGCG:     call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @emit_for_with_vars.omp_outlined, ptr {{.*}})

// LLVM:         call void @after(i32
// LLVM-NEXT:    ret void

// Loop bounds aren't known until runtime (`ub` and `step` are variables, not
// constants), so both paths have to compute the number of iterations on the
// fly. We check the loop setup, during() call (one per iteration), and the
// matching init/fini/barrier call betwen CIR and OGCG.
//
// LLVM-CIR-LABEL: define internal void @emit_for_with_vars..omp_par(
// LLVM-CIR:         store i32 1, ptr %[[LB:.*]], align 4
// LLVM-CIR-NEXT:    store i64 10, ptr %[[UB:.*]], align 8
// LLVM-CIR-NEXT:    store i16 1, ptr %[[STEP:.*]], align 2
// LLVM-CIR:         %{{.*}} = sext i16 %{{.*}} to i32
// LLVM-CIR:         %{{.*}} = sext i32 %{{.*}} to i64
// LLVM-CIR:         %{{.*}} = sext i16 %{{.*}} to i64
// LLVM-CIR:         %{{.*}} = sdiv i64
// LLVM-CIR:         %omp_loop.tripcount = select i1 %{{.*}}, i32 0, i32 %{{.*}}
// LLVM-CIR:         store i32 0, ptr %p.lowerbound, align 4
// LLVM-CIR-NEXT:    %[[TRIPM1:.*]] = sub i32 %omp_loop.tripcount, 1
// LLVM-CIR-NEXT:    store i32 %[[TRIPM1]], ptr %p.upperbound, align 4
// LLVM-CIR-NEXT:    store i32 1, ptr %p.stride, align 4
// LLVM-CIR:         %[[TID3:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-CIR-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 34,
// LLVM-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-CIR-SAME:      i32 1, i32 0)
// LLVM-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-CIR:       omp_loop.exit:
// LLVM-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-CIR:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-CIR:       omp_loop.body:
// LLVM-CIR:         call void @during(i32
// LLVM-CIR:         %omp_loop.next = add nuw i32 %omp_loop.iv, 1
// LLVM-CIR:         ret void

// LLVM-OGCG-LABEL: define internal void @emit_for_with_vars.omp_outlined(
// LLVM-OGCG-SAME:    ptr noalias noundef %.global_tid., ptr noalias noundef %.bound_tid.,
// LLVM-OGCG-SAME:    ptr nofreeobj noundef nonnull align 4 dereferenceable(4) %j)
// LLVM-OGCG:         %lb = alloca i32, align 4
// LLVM-OGCG-NEXT:    %ub = alloca i64, align 8
// LLVM-OGCG-NEXT:    %step = alloca i16, align 2
// LLVM-OGCG:         store i32 1, ptr %lb, align 4
// LLVM-OGCG-NEXT:    store i64 10, ptr %ub, align 8
// LLVM-OGCG-NEXT:    store i16 1, ptr %step, align 2
// LLVM-OGCG:         %{{.*}} = sext i16 %{{.*}} to i32
// LLVM-OGCG:         %{{.*}} = sext i32 %{{.*}} to i64
// LLVM-OGCG:         %{{.*}} = sext i16 %{{.*}} to i64
// LLVM-OGCG:         %{{.*}} = sdiv i64
// LLVM-OGCG:         %[[PRECOND:.*]] = icmp slt i64 0, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[PRECOND]], label %omp.precond.then, label %omp.precond.end
// LLVM-OGCG:       omp.precond.then:
// LLVM-OGCG:         store i32 0, ptr %.omp.lb, align 4
// LLVM-OGCG:         store i32 1, ptr %.omp.stride, align 4
// LLVM-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-OGCG-SAME:      ptr {{.*}}, i32 %{{.*}}, i32 34,
// LLVM-OGCG-SAME:      i32 1, i32 1)
// LLVM-OGCG:         %[[CMP9:.*]] = icmp sgt i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP9]], label %cond.true, label %cond.false
// LLVM-OGCG:       cond.end:
// LLVM-OGCG:         %[[COND2:.*]] = phi i32 [ %{{.*}}, %cond.true ], [ %{{.*}}, %cond.false ]
// LLVM-OGCG-NEXT:    store i32 %[[COND2]], ptr %.omp.ub, align 4
// LLVM-OGCG:       omp.inner.for.cond:
// LLVM-OGCG:         %[[CMP11:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-OGCG-NEXT:    br i1 %[[CMP11]], label %omp.inner.for.body, label %omp.inner.for.end
// LLVM-OGCG:       omp.inner.for.body:
// LLVM-OGCG:         call void @during(i32
// LLVM-OGCG:       omp.loop.exit:
// LLVM-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %{{.*}})
// LLVM-OGCG:       omp.precond.end:
// LLVM-OGCG:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-OGCG-NEXT:    ret void
