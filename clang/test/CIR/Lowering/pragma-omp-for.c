// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --input-file %t-cir.ll

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }
// CHECK: @[[LOC:.*]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
// CHECK: @{{[0-9]+}} = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @[[LOC]] }, align 8
// CHECK: @{{[0-9]+}} = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @[[LOC]] }, align 8

void before(int);
// CHECK: declare void @before(i32)
void during(int);
// CHECK: declare void @during(i32)
void after(int);
// CHECK: declare void @after(i32)

// Test simple for loop with constant bounds: for (int i = 0; i < 10; i++)
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

// CHECK-LABEL: define{{.*}} void @emit_simple_for()
// CHECK:    %structArg = alloca { ptr, ptr }, align 8
// CHECK:    %[[I_ALLOCA:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[J_ALLOCA:.*]] = alloca i32, i64 1, align 4
// CHECK:    store i32 5, ptr %[[J_ALLOCA]], align 4
// CHECK:    %[[J_VAL:.*]] = load i32, ptr %[[J_ALLOCA]], align 4
// CHECK:    call void @before(i32 %[[J_VAL]])
// CHECK:    br label %entry

// CHECK: entry:
// CHECK:    %[[GTN_OUTER:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
// CHECK:    br label %omp_parallel

// CHECK: omp_parallel:
// CHECK:    %[[GEP_I:.*]] = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 0
// CHECK:    store ptr %[[I_ALLOCA]], ptr %[[GEP_I]], align 8
// CHECK:    %[[GEP_J:.*]] = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 1
// CHECK:    store ptr %[[J_ALLOCA]], ptr %[[GEP_J]], align 8
// CHECK:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 1, ptr @emit_simple_for..omp_par, ptr %structArg)
// CHECK:    br label %omp.par.exit

// CHECK: omp.par.exit:
// CHECK:    %[[J_AFTER:.*]] = load i32, ptr %[[J_ALLOCA]], align 4
// CHECK:    call void @after(i32 %[[J_AFTER]])
// CHECK:    ret void


// CHECK-LABEL: define{{.*}} void @emit_simple_for..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0)

// CHECK: omp.par.entry:
// CHECK:    %[[GEP_I_PAR:.*]] = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
// CHECK:    %[[LOADGEP_I:.*]] = load ptr, ptr %[[GEP_I_PAR]], align 8
// CHECK:    %[[GEP_J_PAR:.*]] = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
// CHECK:    %[[LOADGEP_J:.*]] = load ptr, ptr %[[GEP_J_PAR]], align 8
// CHECK:    %p.lastiter = alloca i32, align 4
// CHECK:    %p.lowerbound = alloca i32, align 4
// CHECK:    %p.upperbound = alloca i32, align 4
// CHECK:    %p.stride = alloca i32, align 4
// CHECK:    %[[TID_LOCAL:.*]] = alloca i32, align 4
// CHECK:    %[[TID_LOAD:.*]] = load i32, ptr %tid.addr, align 4
// CHECK:    store i32 %[[TID_LOAD]], ptr %[[TID_LOCAL]], align 4
// CHECK:    %[[TID:.*]] = load i32, ptr %[[TID_LOCAL]], align 4
// CHECK:    br label %omp.region.after_alloca2

// CHECK: omp.region.after_alloca2:
// CHECK:    br label %omp.region.after_alloca

// CHECK: omp.region.after_alloca:
// CHECK:    br label %omp.par.region

// CHECK: omp.par.region:
// CHECK:    br label %omp.par.region1

// CHECK: omp.par.region1:
// initialize i = 0 before the worksharing loop
// CHECK:    store i32 0, ptr %[[LOADGEP_I]], align 4
// CHECK:    br label %omp.wsloop.region

// CHECK: omp.wsloop.region:
// CHECK:    br label %omp_loop.preheader

// CHECK: omp_loop.preheader:
// set normalized loop bounds: lb=0, ub=9 (tripcount-1), stride=1
// CHECK:    store i32 0, ptr %p.lowerbound, align 4
// CHECK:    store i32 9, ptr %p.upperbound, align 4
// CHECK:    store i32 1, ptr %p.stride, align 4
// CHECK:    %[[GTN_WSLOOP:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
// CHECK:    call void @__kmpc_for_static_init_4u(ptr @{{[0-9]+}}, i32 %[[GTN_WSLOOP]], i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
// reload thread-local lb/ub after static partitioning and compute local trip count
// CHECK:    %[[SF_LB:.*]] = load i32, ptr %p.lowerbound, align 4
// CHECK:    %[[SF_UB:.*]] = load i32, ptr %p.upperbound, align 4
// CHECK:    %[[SF_DIFF:.*]] = sub i32 %[[SF_UB]], %[[SF_LB]]
// CHECK:    %[[SF_TC:.*]] = add i32 %[[SF_DIFF]], 1
// CHECK:    br label %omp_loop.header

// CHECK: omp_loop.header:
// CHECK:    %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %[[LOOP_NEXT:.*]], %omp_loop.inc ]
// CHECK:    br label %omp_loop.cond

// CHECK: omp_loop.cond:
// CHECK:    %[[LOOP_CMP:.*]] = icmp ult i32 %omp_loop.iv, %[[SF_TC]]
// CHECK:    br i1 %[[LOOP_CMP]], label %omp_loop.body, label %omp_loop.exit

// CHECK: omp_loop.exit:
// CHECK:    call void @__kmpc_for_static_fini(ptr @{{[0-9]+}}, i32 %[[GTN_WSLOOP]])
// CHECK:    %[[GTN_BARRIER:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
// CHECK:    call void @__kmpc_barrier(ptr @{{[0-9]+}}, i32 %[[GTN_BARRIER]])
// CHECK:    br label %omp_loop.after

// CHECK: omp_loop.after:
// CHECK:    br label %omp.region.cont3

// CHECK: omp.region.cont3:
// CHECK:    br label %omp.region.cont

// CHECK: omp.region.cont:
// CHECK:    br label %omp.par.pre_finalize

// CHECK: omp.par.pre_finalize:
// CHECK:    br label %.fini

// CHECK: .fini:
// CHECK:    br label %omp.par.exit.exitStub

// CHECK: omp_loop.body:
// real IV = (normalized_iv + lb_offset) * stride + init_val
// CHECK:    %[[BODY_IV:.*]] = add i32 %omp_loop.iv, %[[SF_LB]]
// CHECK:    %[[BODY_SCALED:.*]] = mul i32 %[[BODY_IV]], 1
// CHECK:    %[[BODY_FINAL:.*]] = add i32 %[[BODY_SCALED]], 0
// CHECK:    br label %omp.loop_nest.region

// CHECK: omp.loop_nest.region:
// store computed IV to i's alloca; load j and call during(j)
// CHECK:    store i32 %[[BODY_FINAL]], ptr %[[LOADGEP_I]], align 4
// CHECK:    %[[J_DURING:.*]] = load i32, ptr %[[LOADGEP_J]], align 4
// CHECK:    call void @during(i32 %[[J_DURING]])
// CHECK:    br label %omp.region.cont4

// CHECK: omp.region.cont4:
// CHECK:    br label %omp_loop.inc

// CHECK: omp_loop.inc:
// CHECK:    %[[LOOP_NEXT]] = add nuw i32 %omp_loop.iv, 1
// CHECK:    br label %omp_loop.header

// CHECK: omp.par.exit.exitStub:
// CHECK:    ret void

// Test for loop with variable bounds and type conversions
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

// CHECK-LABEL: define{{.*}} void @emit_for_with_vars()
// CHECK:    %structArg = alloca { ptr, ptr, ptr, ptr, ptr }, align 8
// CHECK:    %[[LB_ALLOCA:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[UB_ALLOCA:.*]] = alloca i64, i64 1, align 8
// CHECK:    %[[STEP_ALLOCA:.*]] = alloca i16, i64 1, align 2
// CHECK:    %[[I_ALLOCA:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[J_ALLOCA:.*]] = alloca i32, i64 1, align 4
// CHECK:    store i32 5, ptr %[[J_ALLOCA]], align 4
// CHECK:    %[[J_VAL:.*]] = load i32, ptr %[[J_ALLOCA]], align 4
// CHECK:    call void @before(i32 %[[J_VAL]])
// CHECK:    br label %entry

// CHECK: entry:
// CHECK:    %[[GTN_OUTER:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
// CHECK:    br label %omp_parallel

// CHECK: omp_parallel:
// CHECK:    %[[GEP_LB:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 0
// CHECK:    store ptr %[[LB_ALLOCA]], ptr %[[GEP_LB]], align 8
// CHECK:    %[[GEP_UB:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 1
// CHECK:    store ptr %[[UB_ALLOCA]], ptr %[[GEP_UB]], align 8
// CHECK:    %[[GEP_STEP:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 2
// CHECK:    store ptr %[[STEP_ALLOCA]], ptr %[[GEP_STEP]], align 8
// CHECK:    %[[GEP_I:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 3
// CHECK:    store ptr %[[I_ALLOCA]], ptr %[[GEP_I]], align 8
// CHECK:    %[[GEP_J:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 4
// CHECK:    store ptr %[[J_ALLOCA]], ptr %[[GEP_J]], align 8
// CHECK:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 1, ptr @emit_for_with_vars..omp_par, ptr %structArg)
// CHECK:    br label %omp.par.exit

// CHECK: omp.par.exit:
// CHECK:    %[[J_AFTER:.*]] = load i32, ptr %[[J_ALLOCA]], align 4
// CHECK:    call void @after(i32 %[[J_AFTER]])
// CHECK:    ret void


// CHECK-LABEL: define{{.*}} void @emit_for_with_vars..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0)

// CHECK: omp.par.entry:
// CHECK:    %[[GEP_LB_PAR:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
// CHECK:    %[[LOADGEP_LB:.*]] = load ptr, ptr %[[GEP_LB_PAR]], align 8
// CHECK:    %[[GEP_UB_PAR:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
// CHECK:    %[[LOADGEP_UB:.*]] = load ptr, ptr %[[GEP_UB_PAR]], align 8
// CHECK:    %[[GEP_STEP_PAR:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
// CHECK:    %[[LOADGEP_STEP:.*]] = load ptr, ptr %[[GEP_STEP_PAR]], align 8
// CHECK:    %[[GEP_I_PAR:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
// CHECK:    %[[LOADGEP_I:.*]] = load ptr, ptr %[[GEP_I_PAR]], align 8
// CHECK:    %[[GEP_J_PAR:.*]] = getelementptr { ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
// CHECK:    %[[LOADGEP_J:.*]] = load ptr, ptr %[[GEP_J_PAR]], align 8
// CHECK:    %p.lastiter = alloca i32, align 4
// CHECK:    %p.lowerbound = alloca i32, align 4
// CHECK:    %p.upperbound = alloca i32, align 4
// CHECK:    %p.stride = alloca i32, align 4
// CHECK:    %[[TID_LOCAL:.*]] = alloca i32, align 4
// CHECK:    %[[TID_LOAD:.*]] = load i32, ptr %tid.addr, align 4
// CHECK:    store i32 %[[TID_LOAD]], ptr %[[TID_LOCAL]], align 4
// CHECK:    %[[TID:.*]] = load i32, ptr %[[TID_LOCAL]], align 4
// CHECK:    br label %omp.region.after_alloca2

// CHECK: omp.region.after_alloca2:
// CHECK:    br label %omp.region.after_alloca

// CHECK: omp.region.after_alloca:
// CHECK:    br label %omp.par.region

// CHECK: omp.par.region:
// CHECK:    br label %omp.par.region1

// CHECK: omp.par.region1:
// initialize lb=1, ub=10, step=1 and i=0
// CHECK:    store i32 1, ptr %[[LOADGEP_LB]], align 4
// CHECK:    store i64 10, ptr %[[LOADGEP_UB]], align 8
// CHECK:    store i16 1, ptr %[[LOADGEP_STEP]], align 2
// load ub and step, truncate/extend to i32 for trip count computation
// CHECK:    %[[UB_VAL:.*]] = load i64, ptr %[[LOADGEP_UB]], align 8
// CHECK:    %[[UB_TRUNC:.*]] = trunc i64 %[[UB_VAL]] to i32
// CHECK:    %[[STEP_VAL:.*]] = load i16, ptr %[[LOADGEP_STEP]], align 2
// CHECK:    %[[STEP_SEXT:.*]] = sext i16 %[[STEP_VAL]] to i32
// CHECK:    store i32 0, ptr %[[LOADGEP_I]], align 4
// CHECK:    br label %omp.wsloop.region

// CHECK: omp.wsloop.region:
// compute absolute value of step to normalize direction
// CHECK:    %[[STEP_NEG:.*]] = icmp slt i32 %[[STEP_SEXT]], 0
// CHECK:    %[[STEP_NEG_VAL:.*]] = sub i32 0, %[[STEP_SEXT]]
// CHECK:    %[[STEP_ABS:.*]] = select i1 %[[STEP_NEG]], i32 %[[STEP_NEG_VAL]], i32 %[[STEP_SEXT]]
// select lb/ub based on step direction
// CHECK:    %[[RANGE_LO:.*]] = select i1 %[[STEP_NEG]], i32 %[[UB_TRUNC]], i32 0
// CHECK:    %[[RANGE_HI:.*]] = select i1 %[[STEP_NEG]], i32 0, i32 %[[UB_TRUNC]]
// CHECK:    %[[RANGE_DIFF:.*]] = sub nsw i32 %[[RANGE_HI]], %[[RANGE_LO]]
// CHECK:    %[[RANGE_EMPTY:.*]] = icmp sle i32 %[[RANGE_HI]], %[[RANGE_LO]]
// compute trip count = (diff - 1) / abs(step) + 1
// CHECK:    %[[TC_SUB:.*]] = sub i32 %[[RANGE_DIFF]], 1
// CHECK:    %[[TC_DIV:.*]] = udiv i32 %[[TC_SUB]], %[[STEP_ABS]]
// CHECK:    %[[TC_ADD:.*]] = add i32 %[[TC_DIV]], 1
// CHECK:    %[[TC_ONE:.*]] = icmp ule i32 %[[RANGE_DIFF]], %[[STEP_ABS]]
// CHECK:    %[[TC_CLAMPED:.*]] = select i1 %[[TC_ONE]], i32 1, i32 %[[TC_ADD]]
// CHECK:    %omp_loop.tripcount = select i1 %[[RANGE_EMPTY]], i32 0, i32 %[[TC_CLAMPED]]
// CHECK:    br label %omp_loop.preheader

// CHECK: omp_loop.preheader:
// set normalized loop bounds: lb=0, ub=tripcount-1, stride=1
// CHECK:    store i32 0, ptr %p.lowerbound, align 4
// CHECK:    %[[TC_MINUS1:.*]] = sub i32 %omp_loop.tripcount, 1
// CHECK:    store i32 %[[TC_MINUS1]], ptr %p.upperbound, align 4
// CHECK:    store i32 1, ptr %p.stride, align 4
// CHECK:    %[[GTN_WSLOOP:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
// CHECK:    call void @__kmpc_for_static_init_4u(ptr @{{[0-9]+}}, i32 %[[GTN_WSLOOP]], i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
// reload thread-local lb/ub after static partitioning and compute local trip count
// CHECK:    %[[SF_LB:.*]] = load i32, ptr %p.lowerbound, align 4
// CHECK:    %[[SF_UB:.*]] = load i32, ptr %p.upperbound, align 4
// CHECK:    %[[SF_DIFF:.*]] = sub i32 %[[SF_UB]], %[[SF_LB]]
// CHECK:    %[[SF_TC:.*]] = add i32 %[[SF_DIFF]], 1
// CHECK:    br label %omp_loop.header

// CHECK: omp_loop.header:
// CHECK:    %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %[[LOOP_NEXT:.*]], %omp_loop.inc ]
// CHECK:    br label %omp_loop.cond

// CHECK: omp_loop.cond:
// CHECK:    %[[LOOP_CMP:.*]] = icmp ult i32 %omp_loop.iv, %[[SF_TC]]
// CHECK:    br i1 %[[LOOP_CMP]], label %omp_loop.body, label %omp_loop.exit

// CHECK: omp_loop.exit:
// CHECK:    call void @__kmpc_for_static_fini(ptr @{{[0-9]+}}, i32 %[[GTN_WSLOOP]])
// CHECK:    %[[GTN_BARRIER:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
// CHECK:    call void @__kmpc_barrier(ptr @{{[0-9]+}}, i32 %[[GTN_BARRIER]])
// CHECK:    br label %omp_loop.after

// CHECK: omp_loop.after:
// CHECK:    br label %omp.region.cont3

// CHECK: omp.region.cont3:
// CHECK:    br label %omp.region.cont

// CHECK: omp.region.cont:
// CHECK:    br label %omp.par.pre_finalize

// CHECK: omp.par.pre_finalize:
// CHECK:    br label %.fini

// CHECK: .fini:
// CHECK:    br label %omp.par.exit.exitStub

// CHECK: omp_loop.body:
// real IV = (normalized_iv + lb_offset) * step + init_val
// CHECK:    %[[BODY_IV:.*]] = add i32 %omp_loop.iv, %[[SF_LB]]
// CHECK:    %[[BODY_SCALED:.*]] = mul i32 %[[BODY_IV]], %[[STEP_SEXT]]
// CHECK:    %[[BODY_FINAL:.*]] = add i32 %[[BODY_SCALED]], 0
// CHECK:    br label %omp.loop_nest.region

// CHECK: omp.loop_nest.region:
// store computed IV to i's alloca; load j and call during(j)
// CHECK:    store i32 %[[BODY_FINAL]], ptr %[[LOADGEP_I]], align 4
// CHECK:    %[[J_DURING:.*]] = load i32, ptr %[[LOADGEP_J]], align 4
// CHECK:    call void @during(i32 %[[J_DURING]])
// CHECK:    br label %omp.region.cont4

// CHECK: omp.region.cont4:
// CHECK:    br label %omp_loop.inc

// CHECK: omp_loop.inc:
// CHECK:    %[[LOOP_NEXT]] = add nuw i32 %omp_loop.iv, 1
// CHECK:    br label %omp_loop.header

// CHECK: omp.par.exit.exitStub:
// CHECK:    ret void

