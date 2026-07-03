// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -DOMP52 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
//expected-no-diagnostics
#ifndef HEADER
#define HEADER

#ifdef OMP52
void func4(){
  #pragma omp metadirective when(user = {condition(0)}  \
                 : parallel for) otherwise(parallel)
      for (int i = 0; i < 100; i++)
        ;
  
  #pragma omp metadirective when(user = {condition(1)}  \
                : parallel for) otherwise()
      for (int i = 0; i < 100; i++)
        ;
  }
#endif

// CHECK-LABEL: define dso_local void @_Z5func4v()
// CHECK: entry
// CHECK-NEXT: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 0, ptr @_Z5func4v.omp_outlined)
// CHECK-NEXT: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 0, ptr @_Z5func4v.omp_outlined.1)
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK-LABEL: define internal void @_Z5func4v.omp_outlined
// CHECK-SAME: (ptr noalias noundef [[DOTGLOBAL_TID_:%.*]],
// CHECK-SAME:  ptr noalias noundef [[DOTBOUND_TID_:%.*]])
// CHECK: entry
// CHECK-NEXT:    [[GLOB_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[BOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[GLOB_TID__ADDR]], align 8
// CHECK-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[BOUND_TID__ADDR]], align 8
// CHECK-NEXT:    store i32 0, ptr [[I]], align 4
// CHECK-NEXT:    br label %for.cond
// CHECK:for.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[I]], align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 100
// CHECK-NEXT:    br i1 [[CMP]], label [[FOR_BODY:%.*]], label [[FOR_END:%.*]]
// CHECK:for.body:
// CHECK-NEXT:    br label [[FOR_INC:%.*]]
// CHECK:for.inc:
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[I]], align 4
// CHECK-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP1]], 1
// CHECK-NEXT:    store i32 [[INC]], ptr [[I]], align 4
// CHECK-NEXT:    br label [[FOR_COND:%.*]], !llvm.loop ![[LOOPID:[0-9]+]]
// CHECK:for.end:
// CHECK-NEXT:  ret void
// CHECK-NEXT:}

// CHECK-LABEL: define internal void @_Z5func4v.omp_outlined.1
// CHECK-SAME: (ptr noalias noundef [[DOTGLOBAL_TID_:%.*]],
// CHECK-SAME:  ptr noalias noundef [[DOTBOUND_TID_:%.*]])
// CHECK-NEXT:  entry
// CHECK-NEXT:    [[GLOB_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[BOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK:       [[OMP_IV:%.*]] = alloca i32, align 4
// CHECK-NEXT:  [[TMP:%.*]] = alloca i32, align 4
// CHECK-NEXT:  [[OMP_LB:%.*]] = alloca i32, align 4
// CHECK-NEXT:  [[OMP_UB:%.*]] = alloca i32, align 4
// CHECK-NEXT:  [[OMP_STRIDE:%.*]] = alloca i32, align 4
// CHECK-NEXT:  [[OMP_IS_LAST:%.*]] = alloca i32, align 4
// CHECK-NEXT:  [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[GLOB_TID__ADDR]], align 8
// CHECK-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[BOUND_TID__ADDR]], align 8
// CHECK:       store i32 0, ptr [[OMP_LB:%.*]], align 4
// CHECK-NEXT:  store i32 99, ptr [[OMP_UB:%.*]], align 4
// CHECK-NEXT:  store i32 1, ptr [[OMP_STRIDE:%.*]], align 4
// CHECK-NEXT:  store i32 0, ptr [[OMP_IS_LAST:%.*]], align 4
// CHECK-NEXT:  [[TID_PTR:%.*]] = load ptr, ptr [[GLOBAL_TID_ADDR:%.*]], align 8
// CHECK-NEXT:  [[TID:%.*]] = load i32, ptr [[TID_PTR]], align 4
// CHECK-NEXT:  call void @__kmpc_for_static_init_4(ptr @2, i32 [[TID]], i32 34, ptr [[OMP_IS_LAST]], ptr [[OMP_LB]], ptr [[OMP_UB]], ptr [[OMP_STRIDE]], i32 1, i32 1)
// CHECK:       [[UB_VAL:%.*]] = load i32, ptr [[OMP_UB]], align 4
// CHECK-NEXT:  [[CMP:%.*]] = icmp sgt i32 [[UB_VAL]], 99
// CHECK-NEXT:  br i1 [[CMP]], label %cond.true, label %cond.false
// CHECK:       cond.true:
// CHECK-NEXT:  br label %cond.end
// CHECK:       cond.false:
// CHECK-NEXT:  [[UB_NEW:%.*]] = load i32, ptr [[OMP_UB]], align 4
// CHECK-NEXT:  br label %cond.end
// CHECK:       cond.end:
// CHECK-NEXT:  [[FINAL_UB:%.*]] = phi i32 [ 99, %cond.true ], [ [[UB_NEW]], %cond.false ]
// CHECK-NEXT:  store i32 [[FINAL_UB]], ptr [[OMP_UB]], align 4
// CHECK-NEXT:  [[LB_VAL:%.*]] = load i32, ptr [[OMP_LB]], align 4
// CHECK-NEXT:  store i32 [[LB_VAL]], ptr [[OMP_IV]], align 4
// CHECK-NEXT:  br label %omp.inner.for.cond
// CHECK:       omp.inner.for.cond:
// CHECK-NEXT:  [[IV_VAL:%.*]] = load i32, ptr [[OMP_IV]], align 4
// CHECK-NEXT:  [[UB_FINAL:%.*]] = load i32, ptr [[OMP_UB]], align 4
// CHECK-NEXT:  [[CMP_LOOP:%.*]] = icmp sle i32 [[IV_VAL]], [[UB_FINAL]]
// CHECK-NEXT:  br i1 [[CMP_LOOP]], label %omp.inner.for.body, label %omp.inner.for.end
// CHECK:       omp.inner.for.body:
// CHECK-NEXT:  [[IV_NEW:%.*]] = load i32, ptr [[OMP_IV]], align 4
// CHECK-NEXT:  [[MUL:%.*]] = mul nsw i32 [[IV_NEW]], 1
// CHECK-NEXT:  [[ADD:%.*]] = add nsw i32 0, [[MUL]]
// CHECK-NEXT:  store i32 [[ADD]], ptr [[I]], align 4
// CHECK-NEXT:  br label %omp.body.continue
// CHECK:       omp.inner.for.inc:
// CHECK-NEXT:  [[IV_CURR:%.*]] = load i32, ptr [[OMP_IV]], align 4
// CHECK-NEXT:  [[IV_NEXT:%.*]] = add nsw i32 [[IV_CURR]], 1
// CHECK-NEXT:  store i32 [[IV_NEXT]], ptr [[OMP_IV]], align 4
// CHECK-NEXT:  br label %omp.inner.for.cond
// CHECK:       omp.loop.exit:
// CHECK-NEXT:  call void @__kmpc_for_static_fini(ptr @2, i32 [[TID]])
// CHECK-NEXT:  ret void
// CHECK-NEXT: }

#endif
