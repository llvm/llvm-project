// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv5-unknown-freebsd -std=c11 | FileCheck %s

// Test that we are generating atomicrmw instructions, rather than
// compare-exchange loops for common atomic ops.  This makes a big difference
// on RISC platforms, where the compare-exchange loop becomes a ll/sc pair for
// the load and then another ll/sc in the loop, expanding to about 30
// instructions when it should be only 4.  It has a smaller, but still
// noticeable, impact on platforms like x86 and RISC-V, where there are atomic
// RMW instructions.
//
// We currently emit cmpxchg loops for most operations on _Bools, because
// they're sufficiently rare that it's not worth making sure that the semantics
// are correct.

struct elem;

struct ptr {
    struct elem *ptr;
};
// CHECK-DAG: %struct.ptr = type { %struct.elem* }

struct elem {
    _Atomic(struct ptr) link;
};
// CHECK-DAG: %struct.elem = type { %struct.ptr }

struct ptr object;
// CHECK-DAG: @object = common global %struct.ptr zeroinitializer

// CHECK-DAG: @testStructGlobal = global {{.*}} { i16 1, i16 2, i16 3, i16 4 }
// CHECK-DAG: @testPromotedStructGlobal = global {{.*}} { %{{.*}} { i16 1, i16 2, i16 3 }, [2 x i8] zeroinitializer }


typedef int __attribute__((vector_size(16))) vector;

_Atomic(_Bool) b;
_Atomic(int) i;
_Atomic(long long) l;
_Atomic(short) s;
_Atomic(char*) p;
_Atomic(float) f;
_Atomic(vector) v;

// CHECK: testinc
void testinc(void)
{
  // Special case for suffix bool++, sets to true and returns the old value.
  // CHECK: atomicrmw xchg i8* @b, i8 1 seq_cst
  b++;
  // CHECK: atomicrmw add i32* @i, i32 1 seq_cst
  i++;
  // CHECK: atomicrmw add i64* @l, i64 1 seq_cst
  l++;
  // CHECK: atomicrmw add i16* @s, i16 1 seq_cst
  s++;
  // Prefix increment
  // Special case for bool: set to true and return true
  // CHECK: store atomic i8 1, i8* @b seq_cst, align 1
  ++b;
  // Currently, we have no variant of atomicrmw that returns the new value, so
  // we have to generate an atomic add, which returns the old value, and then a
  // non-atomic add.
  // CHECK: atomicrmw add i32* @i, i32 1 seq_cst
  // CHECK: add i32 
  ++i;
  // CHECK: atomicrmw add i64* @l, i64 1 seq_cst
  // CHECK: add i64
  ++l;
  // CHECK: atomicrmw add i16* @s, i16 1 seq_cst
  // CHECK: add i16
  ++s;
}
// CHECK: testdec
void testdec(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 1, i8* @b
  b--;
  // CHECK: atomicrmw sub i32* @i, i32 1 seq_cst
  i--;
  // CHECK: atomicrmw sub i64* @l, i64 1 seq_cst
  l--;
  // CHECK: atomicrmw sub i16* @s, i16 1 seq_cst
  s--;
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 1, i8* @b
  --b;
  // CHECK: atomicrmw sub i32* @i, i32 1 seq_cst
  // CHECK: sub i32
  --i;
  // CHECK: atomicrmw sub i64* @l, i64 1 seq_cst
  // CHECK: sub i64
  --l;
  // CHECK: atomicrmw sub i16* @s, i16 1 seq_cst
  // CHECK: sub i16
  --s;
}
// CHECK: testaddeq
void testaddeq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 1, i8* @b
  // CHECK: atomicrmw add i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw add i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw add i16* @s, i16 42 seq_cst
  b += 42;
  i += 42;
  l += 42;
  s += 42;
}
// CHECK: testsubeq
void testsubeq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 1, i8* @b
  // CHECK: atomicrmw sub i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw sub i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw sub i16* @s, i16 42 seq_cst
  b -= 42;
  i -= 42;
  l -= 42;
  s -= 42;
}
// CHECK: testxoreq
void testxoreq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 1, i8* @b
  // CHECK: atomicrmw xor i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw xor i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw xor i16* @s, i16 42 seq_cst
  b ^= 42;
  i ^= 42;
  l ^= 42;
  s ^= 42;
}
// CHECK: testoreq
void testoreq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 1, i8* @b
  // CHECK: atomicrmw or i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw or i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw or i16* @s, i16 42 seq_cst
  b |= 42;
  i |= 42;
  l |= 42;
  s |= 42;
}
// CHECK: testandeq
void testandeq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 1, i8* @b
  // CHECK: atomicrmw and i32* @i, i32 42 seq_cst
  // CHECK: atomicrmw and i64* @l, i64 42 seq_cst
  // CHECK: atomicrmw and i16* @s, i16 42 seq_cst
  b &= 42;
  i &= 42;
  l &= 42;
  s &= 42;
}

// CHECK-LABEL: define arm_aapcscc void @testFloat(float*
void testFloat(_Atomic(float) *fp) {
// CHECK:      [[FP:%.*]] = alloca float*
// CHECK-NEXT: [[X:%.*]] = alloca float
// CHECK-NEXT: [[F:%.*]] = alloca float
// CHECK-NEXT: [[TMP0:%.*]] = alloca float
// CHECK-NEXT: [[TMP1:%.*]] = alloca float
// CHECK-NEXT: store float* {{%.*}}, float** [[FP]]

// CHECK-NEXT: [[T0:%.*]] = load float*, float** [[FP]]
// CHECK-NEXT: store float 1.000000e+00, float* [[T0]], align 4
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: store float 2.000000e+00, float* [[X]], align 4
  _Atomic(float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load float*, float** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast float* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast float* [[TMP0]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 4, i8* [[T1]], i8* [[T2]], i32 5)
// CHECK-NEXT: [[T3:%.*]] = load float, float* [[TMP0]], align 4
// CHECK-NEXT: store float [[T3]], float* [[F]]
  float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load float, float* [[F]], align 4
// CHECK-NEXT: [[T1:%.*]] = load float*, float** [[FP]], align 4
// CHECK-NEXT: store float [[T0]], float* [[TMP1]], align 4
// CHECK-NEXT: [[T2:%.*]] = bitcast float* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = bitcast float* [[TMP1]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 4, i8* [[T2]], i8* [[T3]], i32 5)
  *fp = f;

// CHECK-NEXT: ret void
}

// CHECK: define arm_aapcscc void @testComplexFloat([[CF:{ float, float }]]*
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
// CHECK-NEXT: [[T1:%.*]] = bitcast [[CF]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[CF]]* [[TMP0]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 8, i8* [[T1]], i8* [[T2]], i32 5)
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
// CHECK-NEXT: [[T0:%.*]] = bitcast [[CF]]* [[DEST]] to i8*
// CHECK-NEXT: [[T1:%.*]] = bitcast [[CF]]* [[TMP1]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 8, i8* [[T0]], i8* [[T1]], i32 5)
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z, w; } S;
_Atomic S testStructGlobal = (S){1, 2, 3, 4};
// CHECK: define arm_aapcscc void @testStruct([[S:.*]]*
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
// CHECK-NEXT: [[T1:%.*]] = bitcast [[S]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[S]]* [[F]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 8, i8* [[T1]], i8* [[T2]], i32 5)
  S f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load [[S]]*, [[S]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[S]]* [[TMP0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[S]]* [[F]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 [[T1]], i8* align 2 [[T2]], i32 8, i1 false)
// CHECK-NEXT: [[T3:%.*]] = bitcast [[S]]* [[T0]] to i8*
// CHECK-NEXT: [[T4:%.*]] = bitcast [[S]]* [[TMP0]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 8, i8* [[T3]], i8* [[T4]], i32 5)
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z; } PS;
_Atomic PS testPromotedStructGlobal = (PS){1, 2, 3};
// CHECK: define arm_aapcscc void @testPromotedStruct([[APS:.*]]*
void testPromotedStruct(_Atomic(PS) *fp) {
// CHECK:      [[FP:%.*]] = alloca [[APS]]*, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[PS:%.*]], align 2
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[TMP1:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[A:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[TMP2:%.*]] = alloca %struct.PS, align 2
// CHECK-NEXT: [[TMP3:%.*]] = alloca [[APS]], align 8
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
// CHECK-NEXT: [[T1:%.*]] = bitcast [[APS]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[APS]]* [[TMP0]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 8, i8* [[T1]], i8* [[T2]], i32 5)
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
// CHECK-NEXT: [[T4:%.*]] = bitcast [[APS]]* [[T0]] to i8*
// CHECK-NEXT: [[T5:%.*]] = bitcast [[APS]]* [[TMP1]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 8, i8* [[T4]], i8* [[T5]], i32 5)
  *fp = f;

// CHECK-NEXT: [[T0:%.*]] = load [[APS]]*, [[APS]]** [[FP]], align 4
// CHECK-NEXT: [[T1:%.*]] = bitcast [[APS]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[APS]]* [[TMP3]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 8, i8* [[T1]], i8* [[T2]], i32 5)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], [[APS]]* [[TMP3]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = bitcast %struct.PS* [[TMP2]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast %struct.PS* [[T0]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[T1]], i8* align 8 [[T2]], i32 6, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds %struct.PS, %struct.PS* [[TMP2]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = load i16, i16* [[T0]], align 2
// CHECK-NEXT: [[T2:%.*]] = sext i16 [[T1]] to i32
// CHECK-NEXT: store i32 [[T2]], i32* [[A]], align 4
  int a = ((PS)*fp).x;

// CHECK-NEXT: ret void
}

PS test_promoted_load(_Atomic(PS) *addr) {
  // CHECK-LABEL: @test_promoted_load(%struct.PS* noalias sret %agg.result, { %struct.PS, [2 x i8] }* %addr)
  // CHECK:   [[ADDR_ARG:%.*]] = alloca { %struct.PS, [2 x i8] }*, align 4
  // CHECK:   [[ATOMIC_RES:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store { %struct.PS, [2 x i8] }* %addr, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load { %struct.PS, [2 x i8] }*, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   [[ADDR64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ADDR]] to i64*
  // CHECK:   [[ATOMIC_RES64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_RES]] to i64*
  // CHECK:   [[ADDR8:%.*]] = bitcast i64* [[ADDR64]] to i8*
  // CHECK:   [[RES:%.*]] = call arm_aapcscc i64 @__atomic_load_8(i8* [[ADDR8]], i32 5)
  // CHECK:   store i64 [[RES]], i64* [[ATOMIC_RES64]], align 8
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
  // CHECK:   [[ADDR8:%.*]] = bitcast i64* [[ADDR64]] to i8*
  // CHECK:   [[VAL64:%.*]] = load i64, i64* [[ATOMIC_VAL64]], align 2
  // CHECK:   call arm_aapcscc void @__atomic_store_8(i8* [[ADDR8]], i64 [[VAL64]], i32 5)
  __c11_atomic_store(addr, *val, 5);
}

PS test_promoted_exchange(_Atomic(PS) *addr, PS *val) {
  // CHECK-LABEL: @test_promoted_exchange(%struct.PS* noalias sret %agg.result, { %struct.PS, [2 x i8] }* %addr, %struct.PS* %val)
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
  // CHECK:   [[ADDR8:%.*]] = bitcast i64* [[ADDR64]] to i8*
  // CHECK:   [[VAL64:%.*]] = load i64, i64* [[ATOMIC_VAL64]], align 2
  // CHECK:   [[RES:%.*]] = call arm_aapcscc i64 @__atomic_exchange_8(i8* [[ADDR8]], i64 [[VAL64]], i32 5)
  // CHECK:   store i64 [[RES]], i64* [[ATOMIC_RES64]], align 8
  // CHECK:   [[ATOMIC_RES_STRUCT:%.*]] = bitcast i64* [[ATOMIC_RES64]] to %struct.PS*
  // CHECK:   [[AGG_RESULT8:%.*]] = bitcast %struct.PS* %agg.result to i8*
  // CHECK:   [[ATOMIC_RES8:%.*]] = bitcast %struct.PS* [[ATOMIC_RES_STRUCT]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[AGG_RESULT8]], i8* align 8 [[ATOMIC_RES8]], i32 6, i1 false)
  return __c11_atomic_exchange(addr, *val, 5);
}

_Bool test_promoted_cmpxchg(_Atomic(PS) *addr, PS *desired, PS *new) {
  // CHECK-LABEL: i1 @test_promoted_cmpxchg({ %struct.PS, [2 x i8] }* %addr, %struct.PS* %desired, %struct.PS* %new) #0 {
  // CHECK:   [[ADDR_ARG:%.*]] = alloca { %struct.PS, [2 x i8] }*, align 4
  // CHECK:   [[DESIRED_ARG:%.*]] = alloca %struct.PS*, align 4
  // CHECK:   [[NEW_ARG:%.*]] = alloca %struct.PS*, align 4
  // CHECK:   [[NONATOMIC_TMP:%.*]] = alloca %struct.PS, align 2
  // CHECK:   [[ATOMIC_DESIRED:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   [[ATOMIC_NEW:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store { %struct.PS, [2 x i8] }* %addr, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   store %struct.PS* %desired, %struct.PS** [[DESIRED_ARG]], align 4
  // CHECK:   store %struct.PS* %new, %struct.PS** [[NEW_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load { %struct.PS, [2 x i8] }*, { %struct.PS, [2 x i8] }** [[ADDR_ARG]], align 4
  // CHECK:   [[DESIRED:%.*]]= load %struct.PS*, %struct.PS** [[DESIRED_ARG]], align 4
  // CHECK:   [[NEW:%.*]] = load %struct.PS*, %struct.PS** [[NEW_ARG]], align 4
  // CHECK:   [[NONATOMIC_TMP8:%.*]] = bitcast %struct.PS* [[NONATOMIC_TMP]] to i8*
  // CHECK:   [[NEW8:%.*]] = bitcast %struct.PS* [[NEW]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 [[NONATOMIC_TMP8]], i8* align 2 [[NEW8]], i32 6, i1 false)
  // CHECK:   [[ADDR64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ADDR]] to i64*
  // CHECK:   [[ATOMIC_DESIRED8:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_DESIRED]] to i8*
  // CHECK:   [[DESIRED8:%.*]] = bitcast %struct.PS* [[DESIRED]]to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[ATOMIC_DESIRED8]], i8* align 2 [[DESIRED8]], i64 6, i1 false)
  // CHECK:   [[ATOMIC_DESIRED64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_DESIRED]] to i64*
  // CHECK:   [[ATOMIC_NEW8:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_NEW]] to i8*
  // CHECK:   [[NONATOMIC_TMP8:%.*]] = bitcast %struct.PS* [[NONATOMIC_TMP]] to i8*
  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[ATOMIC_NEW8]], i8* align 2 [[NONATOMIC_TMP8]], i64 6, i1 false)
  // CHECK:   [[ATOMIC_NEW64:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[ATOMIC_NEW]] to i64*
  // CHECK:   [[ADDR8:%.*]] = bitcast i64* [[ADDR64]] to i8*
  // CHECK:   [[ATOMIC_DESIRED8:%.*]] = bitcast i64* [[ATOMIC_DESIRED64]] to i8*
  // CHECK:   [[NEW64:%.*]] = load i64, i64* [[ATOMIC_NEW64]], align 2
  // CHECK:   [[RES:%.*]] = call arm_aapcscc zeroext i1 @__atomic_compare_exchange_8(i8* [[ADDR8]], i8* [[ATOMIC_DESIRED8]], i64 [[NEW64]], i32 5, i32 5)
  // CHECK:   ret i1 [[RES]]
  return __c11_atomic_compare_exchange_strong(addr, desired, *new, 5, 5);
}
