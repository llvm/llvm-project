// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

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

// LLVM-LABEL: define{{.*}} void @emit_simple_for()
// LLVM:         %[[STRUCT_ARG:.*]] = alloca { ptr, ptr }, align 8
// LLVM:         %[[I_SLOT:.*]] = alloca i32, i64 1, align 4
// LLVM:         %[[J_SLOT:.*]] = alloca i32, i64 1, align 4
// LLVM:         store i32 5, ptr %[[J_SLOT]], align 4
// LLVM:         call void @before(i32
// LLVM:         %[[GEP0:.*]] = getelementptr { ptr, ptr }, ptr %[[STRUCT_ARG]], i32 0, i32 0
// LLVM-NEXT:    store ptr %[[I_SLOT]], ptr %[[GEP0]], align 8
// LLVM:         %[[GEP1:.*]] = getelementptr { ptr, ptr }, ptr %[[STRUCT_ARG]], i32 0, i32 1
// LLVM-NEXT:    store ptr %[[J_SLOT]], ptr %[[GEP1]], align 8
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(
// LLVM-SAME:      ptr @{{.*}}, i32 1, ptr @emit_simple_for..omp_par, ptr %[[STRUCT_ARG]])
// LLVM:         %[[J_VAL:.*]] = load i32, ptr %[[J_SLOT]], align 4
// LLVM-NEXT:    call void @after(i32 noundef %[[J_VAL]])
// LLVM-NEXT:    ret void

// OGCG-LABEL: define{{.*}} void @emit_simple_for()
// OGCG:         %[[J:.*]] = alloca i32, align 4
// OGCG:         store i32 5, ptr %[[J]], align 4
// OGCG:         %[[J_VAL:.*]] = load i32, ptr %[[J]], align 4
// OGCG-NEXT:    call void @before(i32 noundef %[[J_VAL]])
// OGCG-NEXT:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(
// OGCG-SAME:      ptr @{{.*}}, i32 1, ptr @emit_simple_for.omp_outlined, ptr %[[J]])
// OGCG:         %[[J_VAL2:.*]] = load i32, ptr %[[J]], align 4
// OGCG-NEXT:    call void @after(i32 noundef %[[J_VAL2]])
// OGCG-NEXT:    ret void

// LLVM-LABEL: define internal void @emit_simple_for..omp_par(
// LLVM-SAME:    ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %[[STRUCT:.*]])
// LLVM:         %[[GEP_I:.*]] = getelementptr { ptr, ptr }, ptr %[[STRUCT]], i32 0, i32 0
// LLVM-NEXT:    %[[I_PTR:.*]] = load ptr, ptr %[[GEP_I]], align 8
// LLVM:         %[[GEP_J:.*]] = getelementptr { ptr, ptr }, ptr %[[STRUCT]], i32 0, i32 1
// LLVM-NEXT:    %[[J_PTR:.*]] = load ptr, ptr %[[GEP_J]], align 8
// LLVM:         %p.lastiter = alloca i32, align 4
// LLVM-NEXT:    %p.lowerbound = alloca i32, align 4
// LLVM-NEXT:    %p.upperbound = alloca i32, align 4
// LLVM-NEXT:    %p.stride = alloca i32, align 4
// LLVM:         store i32 0, ptr %p.lowerbound, align 4
// LLVM-NEXT:    store i32 9, ptr %p.upperbound, align 4
// LLVM-NEXT:    store i32 1, ptr %p.stride, align 4
// LLVM:         %[[TID:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
// LLVM-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-SAME:      ptr @{{.*}}, i32 %[[TID]], i32 34,
// LLVM-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-SAME:      i32 1, i32 0)
// LLVM:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM:       omp_loop.exit:
// LLVM-NEXT:    call void @__kmpc_for_static_fini(ptr @{{.*}}, i32 %[[TID]])
// LLVM:         call void @__kmpc_barrier(ptr @{{.*}}, i32 %{{.*}})
// LLVM:       omp_loop.body:
// LLVM:         %[[J_VAL:.*]] = load i32, ptr %[[J_PTR]], align 4
// LLVM-NEXT:    call void @during(i32 noundef %[[J_VAL]])
// LLVM:         %omp_loop.next = add nuw i32 %omp_loop.iv, 1
// LLVM:         ret void

// OGCG-LABEL: define internal void @emit_simple_for.omp_outlined(
// OGCG-SAME:    ptr noalias noundef %.global_tid., ptr noalias noundef %.bound_tid.,
// OGCG-SAME:    ptr noundef nonnull align 4 dereferenceable(4) %j)
// OGCG:         %.global_tid..addr = alloca ptr, align 8
// OGCG-NEXT:    %.bound_tid..addr = alloca ptr, align 8
// OGCG-NEXT:    %j.addr = alloca ptr, align 8
// OGCG:         %.omp.lb = alloca i32, align 4
// OGCG-NEXT:    %.omp.ub = alloca i32, align 4
// OGCG-NEXT:    %.omp.stride = alloca i32, align 4
// OGCG-NEXT:    %.omp.is_last = alloca i32, align 4
// OGCG:         store ptr %.global_tid., ptr %.global_tid..addr, align 8
// OGCG:         store ptr %j, ptr %j.addr, align 8
// OGCG:         %[[J_LOADED:.*]] = load ptr, ptr %j.addr, align 8
// OGCG:         store i32 0, ptr %.omp.lb, align 4
// OGCG-NEXT:    store i32 9, ptr %.omp.ub, align 4
// OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// OGCG:         %[[GTID_VAL_PTR:.*]] = load ptr, ptr %.global_tid..addr, align 8
// OGCG-NEXT:    %[[TID:.*]] = load i32, ptr %[[GTID_VAL_PTR]], align 4
// OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// OGCG-SAME:      ptr @{{.*}}, i32 %[[TID]], i32 34,
// OGCG-SAME:      ptr %.omp.is_last, ptr %.omp.lb, ptr %.omp.ub, ptr %.omp.stride,
// OGCG-SAME:      i32 1, i32 1)
// OGCG:         %[[UB_VAL:.*]] = load i32, ptr %.omp.ub, align 4
// OGCG-NEXT:    %[[CMP:.*]] = icmp sgt i32 %[[UB_VAL]], 9
// OGCG-NEXT:    br i1 %[[CMP]], label %cond.true, label %cond.false
// OGCG:       cond.end:
// OGCG:         %[[COND:.*]] = phi i32 [ 9, %cond.true ], [ %{{.*}}, %cond.false ]
// OGCG-NEXT:    store i32 %[[COND]], ptr %.omp.ub, align 4
// OGCG:       omp.inner.for.cond:
// OGCG:         %[[CMP1:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// OGCG-NEXT:    br i1 %[[CMP1]], label %omp.inner.for.body, label %omp.inner.for.end
// OGCG:       omp.inner.for.body:
// OGCG:         %[[J_VAL:.*]] = load i32, ptr %[[J_LOADED]], align 4
// OGCG-NEXT:    call void @during(i32 noundef %[[J_VAL]])
// OGCG:       omp.loop.exit:
// OGCG:         call void @__kmpc_for_static_fini(ptr @{{.*}}, i32 %[[TID]])
// OGCG-NEXT:    call void @__kmpc_barrier(ptr @{{.*}}, i32 %[[TID]])
// OGCG-NEXT:    ret void

// OGCG: declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)
// OGCG: declare void @__kmpc_for_static_fini(ptr, i32)
// OGCG: declare void @__kmpc_barrier(ptr, i32)
// OGCG: declare {{.*}}void @__kmpc_fork_call(ptr, i32, ptr, ...)

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
// LLVM:         %[[STRUCT_ARG:.*]] = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
// LLVM:         call void @before(i32
// LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(
// LLVM-SAME:      ptr @{{.*}}, i32 1, ptr @emit_for_with_vars..omp_par, ptr %[[STRUCT_ARG]])
// LLVM:         call void @after(i32
// LLVM:         ret void

// OGCG-LABEL: define{{.*}} void @emit_for_with_vars()
// OGCG:         %[[J:.*]] = alloca i32, align 4
// OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(
// OGCG-SAME:      ptr @{{.*}}, i32 1, ptr @emit_for_with_vars.omp_outlined, ptr %[[J]])
// OGCG:         ret void

// LLVM-LABEL: define internal void @emit_for_with_vars..omp_par(
// LLVM-SAME:    ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %[[STRUCT:.*]])
// LLVM:         store i32 1, ptr %{{.*}}, align 4
// LLVM:         store i64 10, ptr %{{.*}}, align 8
// LLVM:         store i16 1, ptr %{{.*}}, align 2
// LLVM:         %{{.*}} = sext i16 %{{.*}} to i32
// LLVM:         %{{.*}} = sext i32 %{{.*}} to i64
// LLVM:         %{{.*}} = sext i16 %{{.*}} to i64
// LLVM:         %{{.*}} = sdiv i64
// LLVM:         %omp_loop.tripcount = select i1 %{{.*}}, i32 0, i32 %{{.*}}
// LLVM:         store i32 0, ptr %p.lowerbound, align 4
// LLVM-NEXT:    %[[UB:.*]] = sub i32 %omp_loop.tripcount, 1
// LLVM-NEXT:    store i32 %[[UB]], ptr %p.upperbound, align 4
// LLVM-NEXT:    store i32 1, ptr %p.stride, align 4
// LLVM:         %[[TID2:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
// LLVM-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-SAME:      ptr @{{.*}}, i32 %[[TID2]], i32 34,
// LLVM-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-SAME:      i32 1, i32 0)
// LLVM:         %omp_loop.iv = phi i32
// LLVM:         icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM:         call void @__kmpc_for_static_fini(ptr @{{.*}}, i32 %[[TID2]])
// LLVM:         call void @__kmpc_barrier(ptr @{{.*}}, i32 %{{.*}})
// LLVM:         call void @during(i32
// LLVM:         ret void

// OGCG-LABEL: define internal void @emit_for_with_vars.omp_outlined(
// OGCG-SAME:    ptr noalias noundef %.global_tid., ptr noalias noundef %.bound_tid.,
// OGCG-SAME:    ptr noundef nonnull align 4 dereferenceable(4) %j)
// OGCG:         %lb = alloca i32, align 4
// OGCG-NEXT:    %ub = alloca i64, align 8
// OGCG-NEXT:    %step = alloca i16, align 2
// OGCG:         store i32 1, ptr %lb, align 4
// OGCG-NEXT:    store i64 10, ptr %ub, align 8
// OGCG-NEXT:    store i16 1, ptr %step, align 2
// OGCG:         %{{.*}} = sext i16 %{{.*}} to i32
// OGCG:         %{{.*}} = sext i32 %{{.*}} to i64
// OGCG:         %{{.*}} = sext i16 %{{.*}} to i64
// OGCG:         %{{.*}} = sdiv i64
// OGCG:         %[[PRECOND:.*]] = icmp slt i64 0, %{{.*}}
// OGCG-NEXT:    br i1 %[[PRECOND]], label %omp.precond.then, label %omp.precond.end
// OGCG:       omp.precond.then:
// OGCG:         store i32 0, ptr %.omp.lb, align 4
// OGCG:         store i32 1, ptr %.omp.stride, align 4
// OGCG:         call void @__kmpc_for_static_init_4(
// OGCG-SAME:      ptr @{{.*}}, i32 %{{.*}}, i32 34,
// OGCG-SAME:      i32 1, i32 1)
// OGCG:         call void @during(i32
// OGCG:         call void @__kmpc_for_static_fini(ptr @{{.*}}, i32 %{{.*}})
// OGCG:       omp.precond.end:
// OGCG:         call void @__kmpc_barrier(ptr @{{.*}}, i32 %{{.*}})
// OGCG-NEXT:    ret void

// LLVM: declare void @__kmpc_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)
// LLVM: declare void @__kmpc_for_static_fini(ptr, i32)
// LLVM: declare i32 @__kmpc_global_thread_num(ptr)
// LLVM: declare void @__kmpc_barrier(ptr, i32)
// LLVM: declare {{.*}}void @__kmpc_fork_call(ptr, i32, ptr, ...)
