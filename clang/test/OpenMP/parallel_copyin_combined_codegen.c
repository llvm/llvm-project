// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#define N 100

int x;
#pragma omp threadprivate(x)

void test_omp_parallel_copyin(int *a) {
  x = 1;

#pragma omp parallel copyin(x)
#pragma omp for
  for (int i = 0; i < N; i++)
    a[i] = i + x;
}

void test_omp_parallel_for_copyin(int *a) {
  x = 2;

#pragma omp parallel for copyin(x)
  for (int i = 0; i < N; i++)
    a[i] = i + x;
}

void test_omp_parallel_for_simd_copyin(int *a) {
  x = 3;

#pragma omp parallel for simd copyin(x)
  for (int i = 0; i < N; i++)
    a[i] = i + x;
}

void test_omp_parallel_sections_copyin(int *a, int *b) {
  x = 4;

#pragma omp parallel sections copyin(x)
  {
#pragma omp section
    { *a = x; }

#pragma omp section
    { *b = x; }
  }
}

void test_omp_parallel_master_copyin(int *a) {
  x = 5;

#pragma omp parallel master copyin(x)
  for (int i = 0; i < N; i++)
    a[i] = i + x;
}

// CHECK-LABEL: define {{[^@]+}}@test_omp_parallel_copyin
// CHECK-SAME: (i32* noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   store i32* [[A]], i32** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   store i32 1, i32* [[TMP0]], align 4
// CHECK-NEXT:   [[TMP1:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32**, i32*)* @.omp_outlined. to void (i32*, i32*, ...)*), i32** [[A_ADDR]], i32* [[TMP1]])
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@.omp_outlined.
// CHECK-SAME: (i32* noalias noundef [[DOTGLOBAL_TID_:%.*]], i32* noalias noundef [[DOTBOUND_TID_:%.*]], i32** noundef nonnull align 8 dereferenceable(8) [[A:%.*]], i32* noundef nonnull align 4 dereferenceable(4) [[X:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[DOTGLOBAL_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTBOUND_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32**, align 8
// CHECK-NEXT:   [[X_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTOMP_IV:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_LB:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_UB:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_STRIDE:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32* [[DOTGLOBAL_TID_]], i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   store i32* [[DOTBOUND_TID_]], i32** [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:   store i32** [[A]], i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   store i32* [[X]], i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = load i32**, i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP1:%.*]] = load i32*, i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP2:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP3:%.*]] = ptrtoint i32* [[TMP1]] to i64
// CHECK-NEXT:   [[TMP4:%.*]] = ptrtoint i32* [[TMP2]] to i64
// CHECK-NEXT:   [[TMP5:%.*]] = icmp ne i64 [[TMP3]], [[TMP4]]
// CHECK-NEXT:   br i1 [[TMP5]], label [[COPYIN_NOT_MASTER:%.*]], label [[COPYIN_NOT_MASTER_END:%.*]]
// CHECK:      copyin.not.master:
// CHECK-NEXT:   [[TMP6:%.*]] = load i32, i32* [[TMP1]], align 4
// CHECK-NEXT:   store i32 [[TMP6]], i32* [[TMP2]], align 4
// CHECK-NEXT:   br label [[COPYIN_NOT_MASTER_END]]
// CHECK:      copyin.not.master.end:
// CHECK-NEXT:   [[TMP7:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP8:%.*]] = load i32, i32* [[TMP7]], align 4
// CHECK-NEXT:   call void @__kmpc_barrier(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP8]])
// CHECK-NEXT:   store i32 0, i32* [[DOTOMP_LB]], align 4
// CHECK-NEXT:   store i32 99, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   store i32 1, i32* [[DOTOMP_STRIDE]], align 4
// CHECK-NEXT:   store i32 0, i32* [[DOTOMP_IS_LAST]], align 4
// CHECK-NEXT:   [[TMP9:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP10:%.*]] = load i32, i32* [[TMP9]], align 4
// CHECK-NEXT:   call void @__kmpc_for_static_init_4(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP10]], i32 34, i32* [[DOTOMP_IS_LAST]], i32* [[DOTOMP_LB]], i32* [[DOTOMP_UB]], i32* [[DOTOMP_STRIDE]], i32 1, i32 1)
// CHECK-NEXT:   [[TMP11:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   [[CMP:%.*]] = icmp sgt i32 [[TMP11]], 99
// CHECK-NEXT:   br i1 [[CMP]], label [[COND_TRUE:%.*]], label [[COND_FALSE:%.*]]
// CHECK:      cond.true:
// CHECK-NEXT:   br label [[COND_END:%.*]]
// CHECK:      cond.false:
// CHECK-NEXT:   [[TMP12:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   br label [[COND_END]]
// CHECK:      cond.end:
// CHECK-NEXT:   [[COND:%.*]] = phi i32 [ 99, [[COND_TRUE]] ], [ [[TMP12]], [[COND_FALSE]] ]
// CHECK-NEXT:   store i32 [[COND]], i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   [[TMP13:%.*]] = load i32, i32* [[DOTOMP_LB]], align 4
// CHECK-NEXT:   store i32 [[TMP13]], i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   br label [[OMP_INNER_FOR_COND:%.*]]
// CHECK:      omp.inner.for.cond:
// CHECK-NEXT:   [[TMP14:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   [[TMP15:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   [[CMP1:%.*]] = icmp sle i32 [[TMP14]], [[TMP15]]
// CHECK-NEXT:   br i1 [[CMP1]], label [[OMP_INNER_FOR_BODY:%.*]], label [[OMP_INNER_FOR_END:%.*]]
// CHECK:      omp.inner.for.body:
// CHECK-NEXT:   [[TMP16:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   [[MUL:%.*]] = mul nsw i32 [[TMP16]], 1
// CHECK-NEXT:   [[ADD:%.*]] = add nsw i32 0, [[MUL]]
// CHECK-NEXT:   store i32 [[ADD]], i32* [[I]], align 4
// CHECK-NEXT:   [[TMP17:%.*]] = load i32, i32* [[I]], align 4
// CHECK-NEXT:   [[TMP18:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP19:%.*]] = load i32, i32* [[TMP18]], align 4
// CHECK-NEXT:   [[ADD2:%.*]] = add nsw i32 [[TMP17]], [[TMP19]]
// CHECK-NEXT:   [[TMP20:%.*]] = load i32*, i32** [[TMP0]], align 8
// CHECK-NEXT:   [[TMP21:%.*]] = load i32, i32* [[I]], align 4
// CHECK-NEXT:   [[IDXPROM:%.*]] = sext i32 [[TMP21]] to i64
// CHECK-NEXT:   [[ARRAYIDX:%.*]] = getelementptr inbounds i32, i32* [[TMP20]], i64 [[IDXPROM]]
// CHECK-NEXT:   store i32 [[ADD2]], i32* [[ARRAYIDX]], align 4
// CHECK-NEXT:   br label [[OMP_BODY_CONTINUE:%.*]]
// CHECK:      omp.body.continue:
// CHECK-NEXT:   br label [[OMP_INNER_FOR_INC:%.*]]
// CHECK:      omp.inner.for.inc:
// CHECK-NEXT:   [[TMP22:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   [[ADD3:%.*]] = add nsw i32 [[TMP22]], 1
// CHECK-NEXT:   store i32 [[ADD3]], i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   br label [[OMP_INNER_FOR_COND]]
// CHECK:      omp.inner.for.end:
// CHECK-NEXT:   br label [[OMP_LOOP_EXIT:%.*]]
// CHECK:      omp.loop.exit:
// CHECK-NEXT:   [[TMP23:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP24:%.*]] = load i32, i32* [[TMP23]], align 4
// CHECK-NEXT:   call void @__kmpc_for_static_fini(%struct.ident_t* @[[GLOB2]], i32 [[TMP24]])
// CHECK-NEXT:   [[TMP25:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP26:%.*]] = load i32, i32* [[TMP25]], align 4
// CHECK-NEXT:   call void @__kmpc_barrier(%struct.ident_t* @[[GLOB1]], i32 [[TMP26]])
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@test_omp_parallel_for_copyin
// CHECK-SAME: (i32* noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   store i32* [[A]], i32** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   store i32 2, i32* [[TMP0]], align 4
// CHECK-NEXT:   [[TMP1:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32**, i32*)* @.omp_outlined..1 to void (i32*, i32*, ...)*), i32** [[A_ADDR]], i32* [[TMP1]])
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@.omp_outlined..1
// CHECK-SAME: (i32* noalias noundef [[DOTGLOBAL_TID_:%.*]], i32* noalias noundef [[DOTBOUND_TID_:%.*]], i32** noundef nonnull align 8 dereferenceable(8) [[A:%.*]], i32* noundef nonnull align 4 dereferenceable(4) [[X:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[DOTGLOBAL_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTBOUND_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32**, align 8
// CHECK-NEXT:   [[X_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTOMP_IV:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_LB:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_UB:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_STRIDE:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32* [[DOTGLOBAL_TID_]], i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   store i32* [[DOTBOUND_TID_]], i32** [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:   store i32** [[A]], i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   store i32* [[X]], i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = load i32**, i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP1:%.*]] = load i32*, i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP2:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP3:%.*]] = ptrtoint i32* [[TMP1]] to i64
// CHECK-NEXT:   [[TMP4:%.*]] = ptrtoint i32* [[TMP2]] to i64
// CHECK-NEXT:   [[TMP5:%.*]] = icmp ne i64 [[TMP3]], [[TMP4]]
// CHECK-NEXT:   br i1 [[TMP5]], label [[COPYIN_NOT_MASTER:%.*]], label [[COPYIN_NOT_MASTER_END:%.*]]
// CHECK:      copyin.not.master:
// CHECK-NEXT:   [[TMP6:%.*]] = load i32, i32* [[TMP1]], align 4
// CHECK-NEXT:   store i32 [[TMP6]], i32* [[TMP2]], align 4
// CHECK-NEXT:   br label [[COPYIN_NOT_MASTER_END]]
// CHECK:      copyin.not.master.end:
// CHECK-NEXT:   [[TMP7:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP8:%.*]] = load i32, i32* [[TMP7]], align 4
// CHECK-NEXT:   call void @__kmpc_barrier(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP8]])
// CHECK-NEXT:   store i32 0, i32* [[DOTOMP_LB]], align 4
// CHECK-NEXT:   store i32 99, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   store i32 1, i32* [[DOTOMP_STRIDE]], align 4
// CHECK-NEXT:   store i32 0, i32* [[DOTOMP_IS_LAST]], align 4
// CHECK-NEXT:   [[TMP9:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP10:%.*]] = load i32, i32* [[TMP9]], align 4
// CHECK-NEXT:   call void @__kmpc_for_static_init_4(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP10]], i32 34, i32* [[DOTOMP_IS_LAST]], i32* [[DOTOMP_LB]], i32* [[DOTOMP_UB]], i32* [[DOTOMP_STRIDE]], i32 1, i32 1)
// CHECK-NEXT:   [[TMP11:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   [[CMP:%.*]] = icmp sgt i32 [[TMP11]], 99
// CHECK-NEXT:   br i1 [[CMP]], label [[COND_TRUE:%.*]], label [[COND_FALSE:%.*]]
// CHECK:      cond.true:
// CHECK-NEXT:   br label [[COND_END:%.*]]
// CHECK:      cond.false:
// CHECK-NEXT:   [[TMP12:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   br label [[COND_END]]
// CHECK:      cond.end:
// CHECK-NEXT:   [[COND:%.*]] = phi i32 [ 99, [[COND_TRUE]] ], [ [[TMP12]], [[COND_FALSE]] ]
// CHECK-NEXT:   store i32 [[COND]], i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   [[TMP13:%.*]] = load i32, i32* [[DOTOMP_LB]], align 4
// CHECK-NEXT:   store i32 [[TMP13]], i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   br label [[OMP_INNER_FOR_COND:%.*]]
// CHECK:      omp.inner.for.cond:
// CHECK-NEXT:   [[TMP14:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   [[TMP15:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   [[CMP1:%.*]] = icmp sle i32 [[TMP14]], [[TMP15]]
// CHECK-NEXT:   br i1 [[CMP1]], label [[OMP_INNER_FOR_BODY:%.*]], label [[OMP_INNER_FOR_END:%.*]]
// CHECK:      omp.inner.for.body:
// CHECK-NEXT:   [[TMP16:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   [[MUL:%.*]] = mul nsw i32 [[TMP16]], 1
// CHECK-NEXT:   [[ADD:%.*]] = add nsw i32 0, [[MUL]]
// CHECK-NEXT:   store i32 [[ADD]], i32* [[I]], align 4
// CHECK-NEXT:   [[TMP17:%.*]] = load i32, i32* [[I]], align 4
// CHECK-NEXT:   [[TMP18:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP19:%.*]] = load i32, i32* [[TMP18]], align 4
// CHECK-NEXT:   [[ADD2:%.*]] = add nsw i32 [[TMP17]], [[TMP19]]
// CHECK-NEXT:   [[TMP20:%.*]] = load i32*, i32** [[TMP0]], align 8
// CHECK-NEXT:   [[TMP21:%.*]] = load i32, i32* [[I]], align 4
// CHECK-NEXT:   [[IDXPROM:%.*]] = sext i32 [[TMP21]] to i64
// CHECK-NEXT:   [[ARRAYIDX:%.*]] = getelementptr inbounds i32, i32* [[TMP20]], i64 [[IDXPROM]]
// CHECK-NEXT:   store i32 [[ADD2]], i32* [[ARRAYIDX]], align 4
// CHECK-NEXT:   br label [[OMP_BODY_CONTINUE:%.*]]
// CHECK:      omp.body.continue:
// CHECK-NEXT:   br label [[OMP_INNER_FOR_INC:%.*]]
// CHECK:      omp.inner.for.inc:
// CHECK-NEXT:   [[TMP22:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   [[ADD3:%.*]] = add nsw i32 [[TMP22]], 1
// CHECK-NEXT:   store i32 [[ADD3]], i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   br label [[OMP_INNER_FOR_COND]]
// CHECK:      omp.inner.for.end:
// CHECK-NEXT:   br label [[OMP_LOOP_EXIT:%.*]]
// CHECK:      omp.loop.exit:
// CHECK-NEXT:   [[TMP23:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP24:%.*]] = load i32, i32* [[TMP23]], align 4
// CHECK-NEXT:   call void @__kmpc_for_static_fini(%struct.ident_t* @[[GLOB2]], i32 [[TMP24]])
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@test_omp_parallel_for_simd_copyin
// CHECK-SAME: (i32* noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   store i32* [[A]], i32** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   store i32 3, i32* [[TMP0]], align 4
// CHECK-NEXT:   [[TMP1:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32**, i32*)* @.omp_outlined..2 to void (i32*, i32*, ...)*), i32** [[A_ADDR]], i32* [[TMP1]])
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@.omp_outlined..2
// CHECK-SAME: (i32* noalias noundef [[DOTGLOBAL_TID_:%.*]], i32* noalias noundef [[DOTBOUND_TID_:%.*]], i32** noundef nonnull align 8 dereferenceable(8) [[A:%.*]], i32* noundef nonnull align 4 dereferenceable(4) [[X:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[DOTGLOBAL_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTBOUND_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32**, align 8
// CHECK-NEXT:   [[X_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTOMP_IV:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_LB:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_UB:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_STRIDE:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32* [[DOTGLOBAL_TID_]], i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   store i32* [[DOTBOUND_TID_]], i32** [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:   store i32** [[A]], i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   store i32* [[X]], i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = load i32**, i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP1:%.*]] = load i32*, i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP2:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP3:%.*]] = ptrtoint i32* [[TMP1]] to i64
// CHECK-NEXT:   [[TMP4:%.*]] = ptrtoint i32* [[TMP2]] to i64
// CHECK-NEXT:   [[TMP5:%.*]] = icmp ne i64 [[TMP3]], [[TMP4]]
// CHECK-NEXT:   br i1 [[TMP5]], label [[COPYIN_NOT_MASTER:%.*]], label [[COPYIN_NOT_MASTER_END:%.*]]
// CHECK:      copyin.not.master:
// CHECK-NEXT:   [[TMP6:%.*]] = load i32, i32* [[TMP1]], align 4
// CHECK-NEXT:   store i32 [[TMP6]], i32* [[TMP2]], align 4
// CHECK-NEXT:   br label [[COPYIN_NOT_MASTER_END]]
// CHECK:      copyin.not.master.end:
// CHECK-NEXT:   [[TMP7:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP8:%.*]] = load i32, i32* [[TMP7]], align 4
// CHECK-NEXT:   call void @__kmpc_barrier(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP8]])
// CHECK-NEXT:   store i32 0, i32* [[DOTOMP_LB]], align 4
// CHECK-NEXT:   store i32 99, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   store i32 1, i32* [[DOTOMP_STRIDE]], align 4
// CHECK-NEXT:   store i32 0, i32* [[DOTOMP_IS_LAST]], align 4
// CHECK-NEXT:   [[TMP9:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP10:%.*]] = load i32, i32* [[TMP9]], align 4
// CHECK-NEXT:   call void @__kmpc_for_static_init_4(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP10]], i32 34, i32* [[DOTOMP_IS_LAST]], i32* [[DOTOMP_LB]], i32* [[DOTOMP_UB]], i32* [[DOTOMP_STRIDE]], i32 1, i32 1)
// CHECK-NEXT:   [[TMP11:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   [[CMP:%.*]] = icmp sgt i32 [[TMP11]], 99
// CHECK-NEXT:   br i1 [[CMP]], label [[COND_TRUE:%.*]], label [[COND_FALSE:%.*]]
// CHECK:      cond.true:
// CHECK-NEXT:   br label [[COND_END:%.*]]
// CHECK:      cond.false:
// CHECK-NEXT:   [[TMP12:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   br label [[COND_END]]
// CHECK:      cond.end:
// CHECK-NEXT:   [[COND:%.*]] = phi i32 [ 99, [[COND_TRUE]] ], [ [[TMP12]], [[COND_FALSE]] ]
// CHECK-NEXT:   store i32 [[COND]], i32* [[DOTOMP_UB]], align 4
// CHECK-NEXT:   [[TMP13:%.*]] = load i32, i32* [[DOTOMP_LB]], align 4
// CHECK-NEXT:   store i32 [[TMP13]], i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   br label [[OMP_INNER_FOR_COND:%.*]]
// CHECK:      omp.inner.for.cond:
// CHECK-NEXT:   [[TMP14:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4, !llvm.access.group
// CHECK-NEXT:   [[TMP15:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4, !llvm.access.group
// CHECK-NEXT:   [[CMP1:%.*]] = icmp sle i32 [[TMP14]], [[TMP15]]
// CHECK-NEXT:   br i1 [[CMP1]], label [[OMP_INNER_FOR_BODY:%.*]], label [[OMP_INNER_FOR_END:%.*]]
// CHECK:      omp.inner.for.body:
// CHECK-NEXT:   [[TMP16:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4, !llvm.access.group
// CHECK-NEXT:   [[MUL:%.*]] = mul nsw i32 [[TMP16]], 1
// CHECK-NEXT:   [[ADD:%.*]] = add nsw i32 0, [[MUL]]
// CHECK-NEXT:   store i32 [[ADD]], i32* [[I]], align 4, !llvm.access.group
// CHECK-NEXT:   [[TMP17:%.*]] = load i32, i32* [[I]], align 4, !llvm.access.group
// CHECK-NEXT:   [[TMP18:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP19:%.*]] = load i32, i32* [[TMP18]], align 4, !llvm.access.group
// CHECK-NEXT:   [[ADD2:%.*]] = add nsw i32 [[TMP17]], [[TMP19]]
// CHECK-NEXT:   [[TMP20:%.*]] = load i32*, i32** [[TMP0]], align 8, !llvm.access.group
// CHECK-NEXT:   [[TMP21:%.*]] = load i32, i32* [[I]], align 4, !llvm.access.group
// CHECK-NEXT:   [[IDXPROM:%.*]] = sext i32 [[TMP21]] to i64
// CHECK-NEXT:   [[ARRAYIDX:%.*]] = getelementptr inbounds i32, i32* [[TMP20]], i64 [[IDXPROM]]
// CHECK-NEXT:   store i32 [[ADD2]], i32* [[ARRAYIDX]], align 4, !llvm.access.group
// CHECK-NEXT:   br label [[OMP_BODY_CONTINUE:%.*]]
// CHECK:      omp.body.continue:
// CHECK-NEXT:   br label [[OMP_INNER_FOR_INC:%.*]]
// CHECK:      omp.inner.for.inc:
// CHECK-NEXT:   [[TMP22:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   [[ADD3:%.*]] = add nsw i32 [[TMP22]], 1
// CHECK-NEXT:   store i32 [[ADD3]], i32* [[DOTOMP_IV]], align 4
// CHECK-NEXT:   br label [[OMP_INNER_FOR_COND]]
// CHECK:      omp.inner.for.end:
// CHECK-NEXT:   br label [[OMP_LOOP_EXIT:%.*]]
// CHECK:      omp.loop.exit:
// CHECK-NEXT:   [[TMP23:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP24:%.*]] = load i32, i32* [[TMP23]], align 4
// CHECK-NEXT:   call void @__kmpc_for_static_fini(%struct.ident_t* @[[GLOB2]], i32 [[TMP24]])
// CHECK-NEXT:   [[TMP25:%.*]] = load i32, i32* [[DOTOMP_IS_LAST]], align 4
// CHECK-NEXT:   [[TMP26:%.*]] = icmp ne i32 [[TMP25]], 0
// CHECK-NEXT:   br i1 [[TMP26]], label [[DOTOMP_FINAL_THEN:%.*]], label [[DOTOMP_FINAL_DONE:%.*]]
// CHECK:      .omp.final.then:
// CHECK-NEXT:   store i32 100, i32* [[I]], align 4
// CHECK-NEXT:   br label [[DOTOMP_FINAL_DONE]]
// CHECK:      .omp.final.done:
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@test_omp_parallel_sections_copyin
// CHECK-SAME: (i32* noundef [[A:%.*]], i32* noundef [[B:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[B_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   store i32* [[A]], i32** [[A_ADDR]], align 8
// CHECK-NEXT:   store i32* [[B]], i32** [[B_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   store i32 4, i32* [[TMP0]], align 4
// CHECK-NEXT:   [[TMP1:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32**, i32**, i32*)* @.omp_outlined..3 to void (i32*, i32*, ...)*), i32** [[A_ADDR]], i32** [[B_ADDR]], i32* [[TMP1]])
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@.omp_outlined..3
// CHECK-SAME: (i32* noalias noundef [[DOTGLOBAL_TID_:%.*]], i32* noalias noundef [[DOTBOUND_TID_:%.*]], i32** noundef nonnull align 8 dereferenceable(8) [[A:%.*]], i32** noundef nonnull align 8 dereferenceable(8) [[B:%.*]], i32* noundef nonnull align 4 dereferenceable(4) [[X:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[DOTGLOBAL_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTBOUND_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32**, align 8
// CHECK-NEXT:   [[B_ADDR:%.*]] = alloca i32**, align 8
// CHECK-NEXT:   [[X_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTOMP_SECTIONS_LB_:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_SECTIONS_UB_:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_SECTIONS_ST_:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_SECTIONS_IL_:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[DOTOMP_SECTIONS_IV_:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32* [[DOTGLOBAL_TID_]], i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   store i32* [[DOTBOUND_TID_]], i32** [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:   store i32** [[A]], i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   store i32** [[B]], i32*** [[B_ADDR]], align 8
// CHECK-NEXT:   store i32* [[X]], i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = load i32**, i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP1:%.*]] = load i32**, i32*** [[B_ADDR]], align 8
// CHECK-NEXT:   [[TMP2:%.*]] = load i32*, i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP3:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP4:%.*]] = ptrtoint i32* [[TMP2]] to i64
// CHECK-NEXT:   [[TMP5:%.*]] = ptrtoint i32* [[TMP3]] to i64
// CHECK-NEXT:   [[TMP6:%.*]] = icmp ne i64 [[TMP4]], [[TMP5]]
// CHECK-NEXT:   br i1 [[TMP6]], label [[COPYIN_NOT_MASTER:%.*]], label [[COPYIN_NOT_MASTER_END:%.*]]
// CHECK:      copyin.not.master:
// CHECK-NEXT:   [[TMP7:%.*]] = load i32, i32* [[TMP2]], align 4
// CHECK-NEXT:   store i32 [[TMP7]], i32* [[TMP3]], align 4
// CHECK-NEXT:   br label [[COPYIN_NOT_MASTER_END]]
// CHECK:      copyin.not.master.end:
// CHECK-NEXT:   [[TMP8:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP9:%.*]] = load i32, i32* [[TMP8]], align 4
// CHECK-NEXT:   call void @__kmpc_barrier(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP9]])
// CHECK-NEXT:   store i32 0, i32* [[DOTOMP_SECTIONS_LB_]], align 4
// CHECK-NEXT:   store i32 1, i32* [[DOTOMP_SECTIONS_UB_]], align 4
// CHECK-NEXT:   store i32 1, i32* [[DOTOMP_SECTIONS_ST_]], align 4
// CHECK-NEXT:   store i32 0, i32* [[DOTOMP_SECTIONS_IL_]], align 4
// CHECK-NEXT:   [[TMP10:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP11:%.*]] = load i32, i32* [[TMP10]], align 4
// CHECK-NEXT:   call void @__kmpc_for_static_init_4(%struct.ident_t* @[[GLOB4:[0-9]+]], i32 [[TMP11]], i32 34, i32* [[DOTOMP_SECTIONS_IL_]], i32* [[DOTOMP_SECTIONS_LB_]], i32* [[DOTOMP_SECTIONS_UB_]], i32* [[DOTOMP_SECTIONS_ST_]], i32 1, i32 1)
// CHECK-NEXT:   [[TMP12:%.*]] = load i32, i32* [[DOTOMP_SECTIONS_UB_]], align 4
// CHECK-NEXT:   [[TMP13:%.*]] = icmp slt i32 [[TMP12]], 1
// CHECK-NEXT:   [[TMP14:%.*]] = select i1 [[TMP13]], i32 [[TMP12]], i32 1
// CHECK-NEXT:   store i32 [[TMP14]], i32* [[DOTOMP_SECTIONS_UB_]], align 4
// CHECK-NEXT:   [[TMP15:%.*]] = load i32, i32* [[DOTOMP_SECTIONS_LB_]], align 4
// CHECK-NEXT:   store i32 [[TMP15]], i32* [[DOTOMP_SECTIONS_IV_]], align 4
// CHECK-NEXT:   br label [[OMP_INNER_FOR_COND:%.*]]
// CHECK:      omp.inner.for.cond:
// CHECK-NEXT:   [[TMP16:%.*]] = load i32, i32* [[DOTOMP_SECTIONS_IV_]], align 4
// CHECK-NEXT:   [[TMP17:%.*]] = load i32, i32* [[DOTOMP_SECTIONS_UB_]], align 4
// CHECK-NEXT:   [[CMP:%.*]] = icmp sle i32 [[TMP16]], [[TMP17]]
// CHECK-NEXT:   br i1 [[CMP]], label [[OMP_INNER_FOR_BODY:%.*]], label [[OMP_INNER_FOR_END:%.*]]
// CHECK:      omp.inner.for.body:
// CHECK-NEXT:   [[TMP18:%.*]] = load i32, i32* [[DOTOMP_SECTIONS_IV_]], align 4
// CHECK-NEXT:   switch i32 [[TMP18]], label [[DOTOMP_SECTIONS_EXIT:%.*]] [
// CHECK-NEXT:     i32 0, label [[DOTOMP_SECTIONS_CASE:%.*]]
// CHECK-NEXT:     i32 1, label [[DOTOMP_SECTIONS_CASE1:%.*]]
// CHECK-NEXT:   ]
// CHECK:      .omp.sections.case:
// CHECK-NEXT:   [[TMP19:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP20:%.*]] = load i32, i32* [[TMP19]], align 4
// CHECK-NEXT:   [[TMP21:%.*]] = load i32*, i32** [[TMP0]], align 8
// CHECK-NEXT:   store i32 [[TMP20]], i32* [[TMP21]], align 4
// CHECK-NEXT:   br label [[DOTOMP_SECTIONS_EXIT]]
// CHECK:      .omp.sections.case1:
// CHECK-NEXT:   [[TMP22:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP23:%.*]] = load i32, i32* [[TMP22]], align 4
// CHECK-NEXT:   [[TMP24:%.*]] = load i32*, i32** [[TMP1]], align 8
// CHECK-NEXT:   store i32 [[TMP23]], i32* [[TMP24]], align 4
// CHECK-NEXT:   br label [[DOTOMP_SECTIONS_EXIT]]
// CHECK:      .omp.sections.exit:
// CHECK-NEXT:   br label [[OMP_INNER_FOR_INC:%.*]]
// CHECK:      omp.inner.for.inc:
// CHECK-NEXT:   [[TMP25:%.*]] = load i32, i32* [[DOTOMP_SECTIONS_IV_]], align 4
// CHECK-NEXT:   [[INC:%.*]] = add nsw i32 [[TMP25]], 1
// CHECK-NEXT:   store i32 [[INC]], i32* [[DOTOMP_SECTIONS_IV_]], align 4
// CHECK-NEXT:   br label [[OMP_INNER_FOR_COND]]
// CHECK:      omp.inner.for.end:
// CHECK-NEXT:   [[TMP26:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP27:%.*]] = load i32, i32* [[TMP26]], align 4
// CHECK-NEXT:   call void @__kmpc_for_static_fini(%struct.ident_t* @[[GLOB4:[0-9]+]], i32 [[TMP27]])
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@test_omp_parallel_master_copyin
// CHECK-SAME: (i32* noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   store i32* [[A]], i32** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   store i32 5, i32* [[TMP0]], align 4
// CHECK-NEXT:   [[TMP1:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32**, i32*)* @.omp_outlined..4 to void (i32*, i32*, ...)*), i32** [[A_ADDR]], i32* [[TMP1]])
// CHECK-NEXT:   ret void
//
// CHECK-LABEL: define {{[^@]+}}@.omp_outlined..4
// CHECK-SAME: (i32* noalias noundef [[DOTGLOBAL_TID_:%.*]], i32* noalias noundef [[DOTBOUND_TID_:%.*]], i32** noundef nonnull align 8 dereferenceable(8) [[A:%.*]], i32* noundef nonnull align 4 dereferenceable(4) [[X:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[DOTGLOBAL_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[DOTBOUND_TID__ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[A_ADDR:%.*]] = alloca i32**, align 8
// CHECK-NEXT:   [[X_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:   [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32* [[DOTGLOBAL_TID_]], i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   store i32* [[DOTBOUND_TID_]], i32** [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:   store i32** [[A]], i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   store i32* [[X]], i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP0:%.*]] = load i32**, i32*** [[A_ADDR]], align 8
// CHECK-NEXT:   [[TMP1:%.*]] = load i32*, i32** [[X_ADDR]], align 8
// CHECK-NEXT:   [[TMP2:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP3:%.*]] = ptrtoint i32* [[TMP1]] to i64
// CHECK-NEXT:   [[TMP4:%.*]] = ptrtoint i32* [[TMP2]] to i64
// CHECK-NEXT:   [[TMP5:%.*]] = icmp ne i64 [[TMP3]], [[TMP4]]
// CHECK-NEXT:   br i1 [[TMP5]], label [[COPYIN_NOT_MASTER:%.*]], label [[COPYIN_NOT_MASTER_END:%.*]]
// CHECK:      copyin.not.master:
// CHECK-NEXT:   [[TMP6:%.*]] = load i32, i32* [[TMP1]], align 4
// CHECK-NEXT:   store i32 [[TMP6]], i32* [[TMP2]], align 4
// CHECK-NEXT:   br label [[COPYIN_NOT_MASTER_END]]
// CHECK:      copyin.not.master.end:
// CHECK-NEXT:   [[TMP7:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP8:%.*]] = load i32, i32* [[TMP7]], align 4
// CHECK-NEXT:   call void @__kmpc_barrier(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP8]])
// CHECK-NEXT:   [[TMP9:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:   [[TMP10:%.*]] = load i32, i32* [[TMP9]], align 4
// CHECK-NEXT:   [[TMP11:%.*]] = call i32 @__kmpc_master(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP10]])
// CHECK-NEXT:   [[TMP12:%.*]] = icmp ne i32 [[TMP11]], 0
// CHECK-NEXT:   br i1 [[TMP12]], label [[OMP_IF_THEN:%.*]], label [[OMP_IF_END:%.*]]
// CHECK:      omp_if.then:
// CHECK-NEXT:   store i32 0, i32* [[I]], align 4
// CHECK-NEXT:   br label [[FOR_COND:%.*]]
// CHECK:      for.cond:
// CHECK-NEXT:   [[TMP13:%.*]] = load i32, i32* [[I]], align 4
// CHECK-NEXT:   [[CMP:%.*]] = icmp slt i32 [[TMP13]], 100
// CHECK-NEXT:   br i1 [[CMP]], label [[FOR_BODY:%.*]], label [[FOR_END:%.*]]
// CHECK:      for.body:
// CHECK-NEXT:   [[TMP14:%.*]] = load i32, i32* [[I]], align 4
// CHECK-NEXT:   [[TMP15:%.*]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @x)
// CHECK-NEXT:   [[TMP16:%.*]] = load i32, i32* [[TMP15]], align 4
// CHECK-NEXT:   [[ADD:%.*]] = add nsw i32 [[TMP14]], [[TMP16]]
// CHECK-NEXT:   [[TMP17:%.*]] = load i32*, i32** [[TMP0]], align 8
// CHECK-NEXT:   [[TMP18:%.*]] = load i32, i32* [[I]], align 4
// CHECK-NEXT:   [[IDXPROM:%.*]] = sext i32 [[TMP18]] to i64
// CHECK-NEXT:   [[ARRAYIDX:%.*]] = getelementptr inbounds i32, i32* [[TMP17]], i64 [[IDXPROM]]
// CHECK-NEXT:   store i32 [[ADD]], i32* [[ARRAYIDX]], align 4
// CHECK-NEXT:   br label [[FOR_INC:%.*]]
// CHECK:      for.inc:
// CHECK-NEXT:   [[TMP19:%.*]] = load i32, i32* [[I]], align 4
// CHECK-NEXT:   [[INC:%.*]] = add nsw i32 [[TMP19]], 1
// CHECK-NEXT:   store i32 [[INC]], i32* [[I]], align 4
// CHECK-NEXT:   br label [[FOR_COND]]
// CHECK:      for.end:
// CHECK-NEXT:   call void @__kmpc_end_master(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP10]])
// CHECK-NEXT:   br label [[OMP_IF_END:%.*]]
// CHECK:      omp_if.end:
// CHECK-NEXT:   ret void
