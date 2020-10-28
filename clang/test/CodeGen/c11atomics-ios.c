// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv7-apple-ios -std=c11 | FileCheck %s

// There isn't really anything special about iOS; it just happens to
// only deploy on processors with native atomics support, so it's a good
// way to test those code-paths.

// This work was done in pursuit of <rdar://13338582>.

// CHECK-LABEL: define void @testFloat(float*
void testFloat(_Atomic(float) *fp) {
// CHECK:      [[FP:%.*]] = alloca float*
// CHECK-NEXT: [[X:%.*]] = alloca float
// CHECK-NEXT: [[F:%.*]] = alloca float
// CHECK-NEXT: store float* {{%.*}}, float** [[FP]]

// CHECK-NEXT: [[T0:%.*]] = load float*, float** [[FP]]
// CHECK-NEXT: store float 1.000000e+00, float* [[T0]], align 4
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: store float 2.000000e+00, float* [[X]], align 4
  _Atomic(float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load float*, float** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast float* [[T0]] to i32*
// CHECK-NEXT: [[T2:%.*]] = load atomic i32, i32* [[T1]] seq_cst, align 4
// CHECK-NEXT: [[T3:%.*]] = bitcast i32 [[T2]] to float
// CHECK-NEXT: store float [[T3]], float* [[F]]
  float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load float, float* [[F]], align 4
// CHECK-NEXT: [[T1:%.*]] = load float*, float** [[FP]], align 4
// CHECK-NEXT: [[T2:%.*]] = bitcast float [[T0]] to i32
// CHECK-NEXT: [[T3:%.*]] = bitcast float* [[T1]] to i32*
// CHECK-NEXT: store atomic i32 [[T2]], i32* [[T3]] seq_cst, align 4
  *fp = f;

// CHECK-NEXT: ret void
}

// CHECK: define void @testComplexFloat([[CF:{ float, float }]]*
void testComplexFloat(_Atomic(_Complex float) *fp) {
// CHECK:      [[FP:%.*]] = alloca [[CF]]*, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[CF]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[CF]], align 4
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[CF]], align 8
// CHECK-NEXT: [[TMP1:%.*]] = alloca [[CF]], align 8
// CHECK-NEXT: store [[CF]]*

// CHECK-NEXT: [[P:%.*]] = load [[CF]]*, [[CF]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[P]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[P]], i32 0, i32 1
// CHECK-NEXT: store float 1.000000e+00, float* [[T0]]
// CHECK-NEXT: store float 0.000000e+00, float* [[T1]]
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[X]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[X]], i32 0, i32 1
// CHECK-NEXT: store float 2.000000e+00, float* [[T0]]
// CHECK-NEXT: store float 0.000000e+00, float* [[T1]]
  _Atomic(_Complex float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load [[CF]]*, [[CF]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[CF]]* [[T0]] to i64*
// CHECK-NEXT: [[T2:%.*]] = load atomic i64, i64* [[T1]] seq_cst, align 8
// CHECK-NEXT: [[T3:%.*]] = bitcast [[CF]]* [[TMP0]] to i64*
// CHECK-NEXT: store i64 [[T2]], i64* [[T3]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[TMP0]], i32 0, i32 0
// CHECK-NEXT: [[R:%.*]] = load float, float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[TMP0]], i32 0, i32 1
// CHECK-NEXT: [[I:%.*]] = load float, float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[F]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[F]], i32 0, i32 1
// CHECK-NEXT: store float [[R]], float* [[T0]]
// CHECK-NEXT: store float [[I]], float* [[T1]]
  _Complex float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[F]], i32 0, i32 0
// CHECK-NEXT: [[R:%.*]] = load float, float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[F]], i32 0, i32 1
// CHECK-NEXT: [[I:%.*]] = load float, float* [[T0]]
// CHECK-NEXT: [[DEST:%.*]] = load [[CF]]*, [[CF]]** [[FP]], align 4
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]], [[CF]]* [[TMP1]], i32 0, i32 1
// CHECK-NEXT: store float [[R]], float* [[T0]]
// CHECK-NEXT: store float [[I]], float* [[T1]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[CF]]* [[TMP1]] to i64*
// CHECK-NEXT: [[T1:%.*]] = load i64, i64* [[T0]], align 8
// CHECK-NEXT: [[T2:%.*]] = bitcast [[CF]]* [[DEST]] to i64*
// CHECK-NEXT: store atomic i64 [[T1]], i64* [[T2]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z, w; } S;
// CHECK: define void @testStruct([[S:.*]]*
void testStruct(_Atomic(S) *fp) {
// CHECK:      [[FP:%.*]] = alloca [[S]]*, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[S]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[S:%.*]], align 2
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[S]], align 8
// CHECK-NEXT: store [[S]]*

// CHECK-NEXT: [[P:%.*]] = load [[S]]*, [[S]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], [[S]]* [[P]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T0]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], [[S]]* [[P]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], [[S]]* [[P]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T0]], align 4
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], [[S]]* [[P]], i32 0, i32 3
// CHECK-NEXT: store i16 4, i16* [[T0]], align 2
  __c11_atomic_init(fp, (S){1,2,3,4});

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], [[S]]* [[X]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T0]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], [[S]]* [[X]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], [[S]]* [[X]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T0]], align 4
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]], [[S]]* [[X]], i32 0, i32 3
// CHECK-NEXT: store i16 4, i16* [[T0]], align 2
  _Atomic(S) x = (S){1,2,3,4};

// CHECK-NEXT: [[T0:%.*]] = load [[S]]*, [[S]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[S]]* [[T0]] to i64*
// CHECK-NEXT: [[T2:%.*]] = load atomic i64, i64* [[T1]] seq_cst, align 8
// CHECK-NEXT: [[T3:%.*]] = bitcast [[S]]* [[F]] to i64*
// CHECK-NEXT: store i64 [[T2]], i64* [[T3]], align 2
  S f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load [[S]]*, [[S]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[S]]* [[TMP0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[S]]* [[F]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 [[T1]], i8* align 2 [[T2]], i32 8, i1 false)
// CHECK-NEXT: [[T3:%.*]] = bitcast [[S]]* [[TMP0]] to i64*
// CHECK-NEXT: [[T4:%.*]] = load i64, i64* [[T3]], align 8
// CHECK-NEXT: [[T5:%.*]] = bitcast [[S]]* [[T0]] to i64*
// CHECK-NEXT: store atomic i64 [[T4]], i64* [[T5]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z; } PS;
// CHECK: define void @testPromotedStruct([[APS:.*]]*
void testPromotedStruct(_Atomic(PS) *fp) {
// CHECK:      [[FP:%.*]] = alloca [[APS]]*, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[PS:%.*]], align 2
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[TMP1:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: store [[APS]]*

// CHECK-NEXT: [[P:%.*]] = load [[APS]]*, [[APS]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[APS]]* [[P]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 8 [[T0]], i8 0, i64 8, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], [[APS]]* [[P]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], [[PS]]* [[T0]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T1]], align 8
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], [[PS]]* [[T0]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], [[PS]]* [[T0]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T1]], align 4
  __c11_atomic_init(fp, (PS){1,2,3});

// CHECK-NEXT: [[T0:%.*]] = bitcast [[APS]]* [[X]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* align 8 [[T0]], i8 0, i32 8, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], [[APS]]* [[X]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], [[PS]]* [[T0]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T1]], align 8
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], [[PS]]* [[T0]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]], [[PS]]* [[T0]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T1]], align 4
  _Atomic(PS) x = (PS){1,2,3};

// CHECK-NEXT: [[T0:%.*]] = load [[APS]]*, [[APS]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[APS]]* [[T0]] to i64*
// CHECK-NEXT: [[T2:%.*]] = load atomic i64, i64* [[T1]] seq_cst, align 8
// CHECK-NEXT: [[T3:%.*]] = bitcast [[APS]]* [[TMP0]] to i64*
// CHECK-NEXT: store i64 [[T2]], i64* [[T3]], align 8
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], [[APS]]* [[TMP0]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = bitcast [[PS]]* [[F]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[PS]]* [[T0]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[T1]], i8* align 8 [[T2]], i32 6, i1 false)
  PS f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load [[APS]]*, [[APS]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[TMP1]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* align 8 [[T1]], i8 0, i32 8, i1 false)
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[APS]], [[APS]]* [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[T2:%.*]] = bitcast [[PS]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = bitcast [[PS]]* [[F]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 [[T2]], i8* align 2 [[T3]], i32 6, i1 false)
// CHECK-NEXT: [[T4:%.*]] = bitcast [[APS]]* [[TMP1]] to i64*
// CHECK-NEXT: [[T5:%.*]] = load i64, i64* [[T4]], align 8
// CHECK-NEXT: [[T6:%.*]] = bitcast [[APS]]* [[T0]] to i64*
// CHECK-NEXT: store atomic i64 [[T5]], i64* [[T6]] seq_cst, align 8
  *fp = f;

// CHECK-NEXT: ret void
}

PS test_promoted_load(_Atomic(PS) *addr) {
  // CHECK-LABEL: @test_promoted_load(%struct.PS* noalias sret(%struct.PS) align 2 %agg.result, { %struct.PS, [2 x i8] }* %addr)
  // CHECK:   [[ADDR_ARG:%.*]] = alloca { %struct.PS, [2 x i8] }*, align 4
  // CHECK:   [[ATOMIC_RES:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store { %struct.PS, [2 x i8] }* %addr, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load { %struct.PS, [2 x i8] }*, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   [[ADDR64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ADDR]] to i64*
  // CHECK:   [[ATOMIC_RES64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_RES]] to i64*
  // CHECK:   [[VAL:%.*]] = load atomic i64, i64* [[ADDR64]] seq_cst, align 8
  // CHECK:   store i64 [[VAL]], i64* [[ATOMIC_RES64]], align 8
  // CHECK:   [[ATOMIC_RES_STRUCT:%.*]] = bitcast i64* [[ATOMIC_RES64]] to %struct.PS*
  // CHECK:   [[AGG_RESULT8:%.*]] = bitcast %struct.PS* %agg.result to i8*
  // CHECK:   [[ATOMIC_RES8:%.*]] = bitcast %struct.PS* [[ATOMIC_RES_STRUCT]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[AGG_RESULT8]], i8* align 8 [[ATOMIC_RES8]], i32 6, i1 false)

  return __c11_atomic_load(addr, 5);
}

void test_promoted_store(_Atomic(PS) *addr, PS *val) {
  // CHECK-LABEL: @test_promoted_store({ %struct.PS, [2 x i8] }* %addr, %struct.PS* %val)
  // CHECK:   [[ADDR_ARG:%.*]] = alloca { %struct.PS, [2 x i8] }*, align 4
  // CHECK:   [[VAL_ARG:%.*]] = alloca %struct.PS*, align 4
  // CHECK:   [[NONATOMIC_TMP:%.*]] = alloca %struct.PS, align 2
  // CHECK:   [[ATOMIC_VAL:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store { %struct.PS, [2 x i8] }* %addr, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   store %struct.PS* %val, %struct.PS** [[VAL_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load { %struct.PS, [2 x i8] }*, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   [[VAL:%.*]] = load %struct.PS*, %struct.PS** [[VAL_ARG]], align 4
  // CHECK:   [[NONATOMIC_TMP8:%.*]] = bitcast %struct.PS* [[NONATOMIC_TMP]] to i8*
  // CHECK:   [[VAL8:%.*]] = bitcast %struct.PS* [[VAL]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[NONATOMIC_TMP8]], i8* align 2 [[VAL8]], i32 6, i1 false)
  // CHECK:   [[ADDR64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ADDR]] to i64*
  // CHECK:   [[ATOMIC_VAL8:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_VAL]] to i8*
  // CHECK:   [[NONATOMIC_TMP8:%.*]] = bitcast %struct.PS* [[NONATOMIC_TMP]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[ATOMIC_VAL8]], i8* align 2 [[NONATOMIC_TMP8]], i64 6, i1 false)
  // CHECK:   [[ATOMIC_VAL64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_VAL]] to i64*
  // CHECK:   [[VAL64:%.*]] = load i64, i64* [[ATOMIC_VAL64]], align 8
  // CHECK:   store atomic i64 [[VAL64]], i64* [[ADDR64]] seq_cst, align 8

  __c11_atomic_store(addr, *val, 5);
}

PS test_promoted_exchange(_Atomic(PS) *addr, PS *val) {
  // CHECK-LABEL: @test_promoted_exchange(%struct.PS* noalias sret(%struct.PS) align 2 %agg.result, { %struct.PS, [2 x i8] }* %addr, %struct.PS* %val)
  // CHECK:   [[ADDR_ARG:%.*]] = alloca { %struct.PS, [2 x i8] }*, align 4
  // CHECK:   [[VAL_ARG:%.*]] = alloca %struct.PS*, align 4
  // CHECK:   [[NONATOMIC_TMP:%.*]] = alloca %struct.PS, align 2
  // CHECK:   [[ATOMIC_VAL:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   [[ATOMIC_RES:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store { %struct.PS, [2 x i8] }* %addr, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   store %struct.PS* %val, %struct.PS** [[VAL_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load { %struct.PS, [2 x i8] }*, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   [[VAL:%.*]] = load %struct.PS*, %struct.PS** [[VAL_ARG]], align 4
  // CHECK:   [[NONATOMIC_TMP8:%.*]] = bitcast %struct.PS* [[NONATOMIC_TMP]] to i8*
  // CHECK:   [[VAL8:%.*]] = bitcast %struct.PS* [[VAL]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[NONATOMIC_TMP8]], i8* align 2 [[VAL8]], i32 6, i1 false)
  // CHECK:   [[ADDR64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ADDR]] to i64*
  // CHECK:   [[ATOMIC_VAL8:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_VAL]] to i8*
  // CHECK:   [[NONATOMIC_TMP8:%.*]] = bitcast %struct.PS* [[NONATOMIC_TMP]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[ATOMIC_VAL8]], i8* align 2 [[NONATOMIC_TMP8]], i64 6, i1 false)
  // CHECK:   [[ATOMIC_VAL64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_VAL]] to i64*
  // CHECK:   [[ATOMIC_RES64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_RES]] to i64*
  // CHECK:   [[VAL64:%.*]] = load i64, i64* [[ATOMIC_VAL64]], align 8
  // CHECK:   [[RES:%.*]] = atomicrmw xchg i64* [[ADDR64]], i64 [[VAL64]] seq_cst
  // CHECK:   store i64 [[RES]], i64* [[ATOMIC_RES64]], align 8
  // CHECK:   [[ATOMIC_RES_STRUCT:%.*]] = bitcast i64* [[ATOMIC_RES64]] to %struct.PS*
  // CHECK:   [[AGG_RESULT8:%.*]] = bitcast %struct.PS* %agg.result to i8*
  // CHECK:   [[ATOMIC_RES8:%.*]] = bitcast %struct.PS* [[ATOMIC_RES_STRUCT]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[AGG_RESULT8]], i8* align 8 [[ATOMIC_RES8]], i32 6, i1 false)
  return __c11_atomic_exchange(addr, *val, 5);
}

_Bool test_promoted_cmpxchg(_Atomic(PS) *addr, PS *desired, PS *new) {
  // CHECK:   define zeroext i1 @test_promoted_cmpxchg({ %struct.PS, [2 x i8] }* %addr, %struct.PS* %desired, %struct.PS* %new) #0 {
  // CHECK:   [[ADDR_ARG:%.*]] = alloca { %struct.PS, [2 x i8] }*, align 4
  // CHECK:   [[DESIRED_ARG:%.*]] = alloca %struct.PS*, align 4
  // CHECK:   [[NEW_ARG:%.*]] = alloca %struct.PS*, align 4
  // CHECK:   [[NONATOMIC_TMP:%.*]] = alloca %struct.PS, align 2
  // CHECK:   [[ATOMIC_DESIRED:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   [[ATOMIC_NEW:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   [[RES_ADDR:%.*]] = alloca i8, align 1
  // CHECK:   store { %struct.PS, [2 x i8] }* %addr, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   store %struct.PS* %desired, %struct.PS** [[DESIRED_ARG]], align 4
  // CHECK:   store %struct.PS* %new, %struct.PS** [[NEW_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load { %struct.PS, [2 x i8] }*, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   [[DESIRED:%.*]] = load %struct.PS*, %struct.PS** [[DESIRED_ARG]], align 4
  // CHECK:   [[NEW:%.*]] = load %struct.PS*, %struct.PS** [[NEW_ARG]], align 4
  // CHECK:   [[NONATOMIC_TMP8:%.*]] = bitcast %struct.PS* [[NONATOMIC_TMP]] to i8*
  // CHECK:   [[NEW8:%.*]] = bitcast %struct.PS* [[NEW]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[NONATOMIC_TMP8]], i8* align 2 [[NEW8]], i32 6, i1 false)
  // CHECK:   [[ADDR64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ADDR]] to i64*
  // CHECK:   [[ATOMIC_DESIRED8:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_DESIRED:%.*]] to i8*
  // CHECK:   [[DESIRED8:%.*]] = bitcast %struct.PS* [[DESIRED]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[ATOMIC_DESIRED8]], i8* align 2 [[DESIRED8]], i64 6, i1 false)
  // CHECK:   [[ATOMIC_DESIRED64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_DESIRED:%.*]] to i64*
  // CHECK:   [[ATOMIC_NEW8:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_NEW]] to i8*
  // CHECK:   [[NONATOMIC_TMP8:%.*]] = bitcast %struct.PS* [[NONATOMIC_TMP]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[ATOMIC_NEW8]], i8* align 2 [[NONATOMIC_TMP8]], i64 6, i1 false)
  // CHECK:   [[ATOMIC_NEW64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_NEW]] to i64*
  // CHECK:   [[ATOMIC_DESIRED_VAL64:%.*]] = load i64, i64* [[ATOMIC_DESIRED64]], align 8
  // CHECK:   [[ATOMIC_NEW_VAL64:%.*]] = load i64, i64* [[ATOMIC_NEW64]], align 8
  // CHECK:   [[RES:%.*]] = cmpxchg i64* [[ADDR64]], i64 [[ATOMIC_DESIRED_VAL64]], i64 [[ATOMIC_NEW_VAL64]] seq_cst seq_cst
  // CHECK:   [[RES_VAL64:%.*]] = extractvalue { i64, i1 } [[RES]], 0
  // CHECK:   [[RES_BOOL:%.*]] = extractvalue { i64, i1 } [[RES]], 1
  // CHECK:   br i1 [[RES_BOOL]], label {{%.*}}, label {{%.*}}

  // CHECK:   store i64 [[RES_VAL64]], i64* [[ATOMIC_DESIRED64]], align 8
  // CHECK:   br label {{%.*}}

  // CHECK:   [[RES_BOOL8:%.*]] = zext i1 [[RES_BOOL]] to i8
  // CHECK:   store i8 [[RES_BOOL8]], i8* [[RES_ADDR]], align 1
  // CHECK:   [[RES_BOOL8:%.*]] = load i8, i8* [[RES_ADDR]], align 1
  // CHECK:   [[RETVAL:%.*]] = trunc i8 [[RES_BOOL8]] to i1
  // CHECK:   ret i1 [[RETVAL]]

  return __c11_atomic_compare_exchange_strong(addr, desired, *new, 5, 5);
}

struct Empty {};

struct Empty testEmptyStructLoad(_Atomic(struct Empty)* empty) {
  // CHECK-LABEL: @testEmptyStructLoad(
  // CHECK-NOT: @__atomic_load
  // CHECK: load atomic i8, i8* %{{.*}} seq_cst, align 1
  return *empty;
}

void testEmptyStructStore(_Atomic(struct Empty)* empty, struct Empty value) {
  // CHECK-LABEL: @testEmptyStructStore(
  // CHECK-NOT: @__atomic_store
  // CHECK: store atomic i8 %{{.*}}, i8* %{{.*}} seq_cst, align 1
  *empty = value;
}
