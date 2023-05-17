// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv7-apple-ios -std=c11 | FileCheck %s

// There isn't really anything special about iOS; it just happens to
// only deploy on processors with native atomics support, so it's a good
// way to test those code-paths.

// This work was done in pursuit of <rdar://13338582>.

// CHECK-LABEL: define{{.*}} void @testFloat(ptr
void testFloat(_Atomic(float) *fp) {
// CHECK:      [[FP:%.*]] = alloca ptr
// CHECK-NEXT: [[X:%.*]] = alloca float
// CHECK-NEXT: [[F:%.*]] = alloca float
// CHECK-NEXT: store ptr {{%.*}}, ptr [[FP]]

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: store float 1.000000e+00, ptr [[T0]], align 4
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: store float 2.000000e+00, ptr [[X]], align 4
  _Atomic(float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: [[T2:%.*]] = load atomic i32, ptr [[T0]] seq_cst, align 4
// CHECK-NEXT: [[T3:%.*]] = bitcast i32 [[T2]] to float
// CHECK-NEXT: store float [[T3]], ptr [[F]]
  float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load float, ptr [[F]], align 4
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[FP]], align 4
// CHECK-NEXT: [[T2:%.*]] = bitcast float [[T0]] to i32
// CHECK-NEXT: store atomic i32 [[T2]], ptr [[T1]] seq_cst, align 4
  *fp = f;

// CHECK-NEXT: ret void
}

// CHECK: define{{.*}} void @testComplexFloat(ptr
void testComplexFloat(_Atomic(_Complex float) *fp) {
// CHECK:      [[FP:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[CF:{ float, float }]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[CF]], align 4
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[CF]], align 8
// CHECK-NEXT: [[TMP1:%.*]] = alloca [[CF]], align 8
// CHECK-NEXT: store ptr

// CHECK-NEXT: [[P:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], ptr [[P]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]], ptr [[P]], i32 0, i32 1
// CHECK-NEXT: store float 1.000000e+00, ptr [[T0]]
// CHECK-NEXT: store float 0.000000e+00, ptr [[T1]]
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], ptr [[X]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]], ptr [[X]], i32 0, i32 1
// CHECK-NEXT: store float 2.000000e+00, ptr [[T0]]
// CHECK-NEXT: store float 0.000000e+00, ptr [[T1]]
  _Atomic(_Complex float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: [[T2:%.*]] = load atomic i64, ptr [[T0]] seq_cst, align 8
// CHECK-NEXT: store i64 [[T2]], ptr [[TMP0]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], ptr [[TMP0]], i32 0, i32 0
// CHECK-NEXT: [[R:%.*]] = load float, ptr [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], ptr [[TMP0]], i32 0, i32 1
// CHECK-NEXT: [[I:%.*]] = load float, ptr [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], ptr [[F]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]], ptr [[F]], i32 0, i32 1
// CHECK-NEXT: store float [[R]], ptr [[T0]]
// CHECK-NEXT: store float [[I]], ptr [[T1]]
  _Complex float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], ptr [[F]], i32 0, i32 0
// CHECK-NEXT: [[R:%.*]] = load float, ptr [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], ptr [[F]], i32 0, i32 1
// CHECK-NEXT: [[I:%.*]] = load float, ptr [[T0]]
// CHECK-NEXT: [[DEST:%.*]] = load ptr, ptr [[FP]], align 4
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]], ptr [[TMP1]], i32 0, i32 1
// CHECK-NEXT: store float [[R]], ptr [[T0]]
// CHECK-NEXT: store float [[I]], ptr [[T1]]
// CHECK-NEXT: [[T1:%.*]] = load i64, ptr [[TMP1]], align 8
// CHECK-NEXT: store atomic i64 [[T1]], ptr [[DEST]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z, w; } S;
// CHECK: define{{.*}} void @testStruct(ptr
void testStruct(_Atomic(S) *fp) {
// CHECK:      [[FP:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[S:.*]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[S:%.*]], align 2
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[S]], align 8
// CHECK-NEXT: store ptr

// CHECK-NEXT: [[P:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], ptr [[P]], i32 0, i32 0
// CHECK-NEXT: store i16 1, ptr [[T0]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], ptr [[P]], i32 0, i32 1
// CHECK-NEXT: store i16 2, ptr [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], ptr [[P]], i32 0, i32 2
// CHECK-NEXT: store i16 3, ptr [[T0]], align 4
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], ptr [[P]], i32 0, i32 3
// CHECK-NEXT: store i16 4, ptr [[T0]], align 2
  __c11_atomic_init(fp, (S){1,2,3,4});

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], ptr [[X]], i32 0, i32 0
// CHECK-NEXT: store i16 1, ptr [[T0]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], ptr [[X]], i32 0, i32 1
// CHECK-NEXT: store i16 2, ptr [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], ptr [[X]], i32 0, i32 2
// CHECK-NEXT: store i16 3, ptr [[T0]], align 4
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], ptr [[X]], i32 0, i32 3
// CHECK-NEXT: store i16 4, ptr [[T0]], align 2
  _Atomic(S) x = (S){1,2,3,4};

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: [[T2:%.*]] = load atomic i64, ptr [[T0]] seq_cst, align 8
// CHECK-NEXT: store i64 [[T2]], ptr [[F]], align 2
  S f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[TMP0]], ptr align 2 [[F]], i32 8, i1 false)
// CHECK-NEXT: [[T4:%.*]] = load i64, ptr [[TMP0]], align 8
// CHECK-NEXT: store atomic i64 [[T4]], ptr [[T0]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z; } PS;
// CHECK: define{{.*}} void @testPromotedStruct(ptr
void testPromotedStruct(_Atomic(PS) *fp) {
// CHECK:      [[FP:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[APS:.*]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[PS:%.*]], align 2
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[TMP1:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: store ptr

// CHECK-NEXT: [[P:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 8 [[P]], i8 0, i64 8, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], ptr [[P]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], ptr [[T0]], i32 0, i32 0
// CHECK-NEXT: store i16 1, ptr [[T1]], align 8
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], ptr [[T0]], i32 0, i32 1
// CHECK-NEXT: store i16 2, ptr [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], ptr [[T0]], i32 0, i32 2
// CHECK-NEXT: store i16 3, ptr [[T1]], align 4
  __c11_atomic_init(fp, (PS){1,2,3});

// CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 8 [[X]], i8 0, i32 8, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], ptr [[X]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], ptr [[T0]], i32 0, i32 0
// CHECK-NEXT: store i16 1, ptr [[T1]], align 8
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], ptr [[T0]], i32 0, i32 1
// CHECK-NEXT: store i16 2, ptr [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], ptr [[T0]], i32 0, i32 2
// CHECK-NEXT: store i16 3, ptr [[T1]], align 4
  _Atomic(PS) x = (PS){1,2,3};

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: [[T2:%.*]] = load atomic i64, ptr [[T0]] seq_cst, align 8
// CHECK-NEXT: store i64 [[T2]], ptr [[TMP0]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], ptr [[TMP0]], i32 0, i32 0
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[F]], ptr align 8 [[T0]], i32 6, i1 false)
  PS f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 8 [[TMP1]], i8 0, i32 8, i1 false)
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[APS]], ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[T1]], ptr align 2 [[F]], i32 6, i1 false)
// CHECK-NEXT: [[T5:%.*]] = load i64, ptr [[TMP1]], align 8
// CHECK-NEXT: store atomic i64 [[T5]], ptr [[T0]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

PS test_promoted_load(_Atomic(PS) *addr) {
  // CHECK-LABEL: @test_promoted_load(ptr noalias sret(%struct.PS) align 2 %agg.result, ptr noundef %addr)
  // CHECK:   [[ADDR_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[ATOMIC_RES:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store ptr %addr, ptr [[ADDR_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load ptr, ptr [[ADDR_ARG]], align 4
  // CHECK:   [[VAL:%.*]] = load atomic i64, ptr [[ADDR]] seq_cst, align 8
  // CHECK:   store i64 [[VAL]], ptr [[ATOMIC_RES]], align 8
  // CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 2 %agg.result, ptr align 8 [[ATOMIC_RES]], i32 6, i1 false)

  return __c11_atomic_load(addr, 5);
}

void test_promoted_store(_Atomic(PS) *addr, PS *val) {
  // CHECK-LABEL: @test_promoted_store(ptr noundef %addr, ptr noundef %val)
  // CHECK:   [[ADDR_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[VAL_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[NONATOMIC_TMP:%.*]] = alloca %struct.PS, align 2
  // CHECK:   [[ATOMIC_VAL:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store ptr %addr, ptr [[ADDR_ARG]], align 4
  // CHECK:   store ptr %val, ptr [[VAL_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load ptr, ptr [[ADDR_ARG]], align 4
  // CHECK:   [[VAL:%.*]] = load ptr, ptr [[VAL_ARG]], align 4
  // CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[NONATOMIC_TMP]], ptr align 2 [[VAL]], i32 6, i1 false)
  // CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ATOMIC_VAL]], ptr align 2 [[NONATOMIC_TMP]], i64 6, i1 false)
  // CHECK:   [[VAL64:%.*]] = load i64, ptr [[ATOMIC_VAL]], align 8
  // CHECK:   store atomic i64 [[VAL64]], ptr [[ADDR]] seq_cst, align 8

  __c11_atomic_store(addr, *val, 5);
}

PS test_promoted_exchange(_Atomic(PS) *addr, PS *val) {
  // CHECK-LABEL: @test_promoted_exchange(ptr noalias sret(%struct.PS) align 2 %agg.result, ptr noundef %addr, ptr noundef %val)
  // CHECK:   [[ADDR_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[VAL_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[NONATOMIC_TMP:%.*]] = alloca %struct.PS, align 2
  // CHECK:   [[ATOMIC_VAL:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   [[ATOMIC_RES:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store ptr %addr, ptr [[ADDR_ARG]], align 4
  // CHECK:   store ptr %val, ptr [[VAL_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load ptr, ptr [[ADDR_ARG]], align 4
  // CHECK:   [[VAL:%.*]] = load ptr, ptr [[VAL_ARG]], align 4
  // CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[NONATOMIC_TMP]], ptr align 2 [[VAL]], i32 6, i1 false)
  // CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ATOMIC_VAL]], ptr align 2 [[NONATOMIC_TMP]], i64 6, i1 false)
  // CHECK:   [[VAL64:%.*]] = load i64, ptr [[ATOMIC_VAL]], align 8
  // CHECK:   [[RES:%.*]] = atomicrmw xchg ptr [[ADDR]], i64 [[VAL64]] seq_cst, align 8
  // CHECK:   store i64 [[RES]], ptr [[ATOMIC_RES]], align 8
  // CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 2 %agg.result, ptr align 8 [[ATOMIC_RES]], i32 6, i1 false)
  return __c11_atomic_exchange(addr, *val, 5);
}

_Bool test_promoted_cmpxchg(_Atomic(PS) *addr, PS *desired, PS *new) {
  // CHECK:   define{{.*}} zeroext i1 @test_promoted_cmpxchg(ptr noundef %addr, ptr noundef %desired, ptr noundef %new) #0 {
  // CHECK:   [[ADDR_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[DESIRED_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[NEW_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[NONATOMIC_TMP:%.*]] = alloca %struct.PS, align 2
  // CHECK:   [[ATOMIC_DESIRED:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   [[ATOMIC_NEW:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   [[RES_ADDR:%.*]] = alloca i8, align 1
  // CHECK:   store ptr %addr, ptr [[ADDR_ARG]], align 4
  // CHECK:   store ptr %desired, ptr [[DESIRED_ARG]], align 4
  // CHECK:   store ptr %new, ptr [[NEW_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load ptr, ptr [[ADDR_ARG]], align 4
  // CHECK:   [[DESIRED:%.*]] = load ptr, ptr [[DESIRED_ARG]], align 4
  // CHECK:   [[NEW:%.*]] = load ptr, ptr [[NEW_ARG]], align 4
  // CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[NONATOMIC_TMP]], ptr align 2 [[NEW]], i32 6, i1 false)
  // CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ATOMIC_DESIRED:%.*]], ptr align 2 [[DESIRED]], i64 6, i1 false)
  // CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ATOMIC_NEW]], ptr align 2 [[NONATOMIC_TMP]], i64 6, i1 false)
  // CHECK:   [[ATOMIC_DESIRED_VAL64:%.*]] = load i64, ptr [[ATOMIC_DESIRED:%.*]], align 8
  // CHECK:   [[ATOMIC_NEW_VAL64:%.*]] = load i64, ptr [[ATOMIC_NEW]], align 8
  // CHECK:   [[RES:%.*]] = cmpxchg ptr [[ADDR]], i64 [[ATOMIC_DESIRED_VAL64]], i64 [[ATOMIC_NEW_VAL64]] seq_cst seq_cst, align 8
  // CHECK:   [[RES_VAL64:%.*]] = extractvalue { i64, i1 } [[RES]], 0
  // CHECK:   [[RES_BOOL:%.*]] = extractvalue { i64, i1 } [[RES]], 1
  // CHECK:   br i1 [[RES_BOOL]], label {{%.*}}, label {{%.*}}

  // CHECK:   store i64 [[RES_VAL64]], ptr [[ATOMIC_DESIRED]], align 8
  // CHECK:   br label {{%.*}}

  // CHECK:   [[RES_BOOL8:%.*]] = zext i1 [[RES_BOOL]] to i8
  // CHECK:   store i8 [[RES_BOOL8]], ptr [[RES_ADDR]], align 1
  // CHECK:   [[RES_BOOL8:%.*]] = load i8, ptr [[RES_ADDR]], align 1
  // CHECK:   [[RETVAL:%.*]] = trunc i8 [[RES_BOOL8]] to i1
  // CHECK:   ret i1 [[RETVAL]]

  return __c11_atomic_compare_exchange_strong(addr, desired, *new, 5, 5);
}

struct Empty {};

struct Empty testEmptyStructLoad(_Atomic(struct Empty)* empty) {
  // CHECK-LABEL: @testEmptyStructLoad(
  // CHECK-NOT: @__atomic_load
  // CHECK: load atomic i8, ptr %{{.*}} seq_cst, align 1
  return *empty;
}

void testEmptyStructStore(_Atomic(struct Empty)* empty, struct Empty value) {
  // CHECK-LABEL: @testEmptyStructStore(
  // CHECK-NOT: @__atomic_store
  // CHECK: store atomic i8 %{{.*}}, ptr %{{.*}} seq_cst, align 1
  *empty = value;
}
