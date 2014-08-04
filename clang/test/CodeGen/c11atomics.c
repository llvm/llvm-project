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

// CHECK: @testStructGlobal = global {{.*}} { i16 1, i16 2, i16 3, i16 4 }
// CHECK: @testPromotedStructGlobal = global {{.*}} { %{{.*}} { i16 1, i16 2, i16 3 }, [2 x i8] zeroinitializer }

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
  // CHECK: cmpxchg i8* @b
  b--;
  // CHECK: atomicrmw sub i32* @i, i32 1 seq_cst
  i--;
  // CHECK: atomicrmw sub i64* @l, i64 1 seq_cst
  l--;
  // CHECK: atomicrmw sub i16* @s, i16 1 seq_cst
  s--;
  // CHECK: cmpxchg i8* @b
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
  // CHECK: cmpxchg i8* @b
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
  // CHECK: cmpxchg i8* @b
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
  // CHECK: cmpxchg i8* @b
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
  // CHECK: cmpxchg i8* @b
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
  // CHECK: cmpxchg i8* @b
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

// CHECK-NEXT: [[T0:%.*]] = load float** [[FP]]
// CHECK-NEXT: store float 1.000000e+00, float* [[T0]], align 4
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: store float 2.000000e+00, float* [[X]], align 4
  _Atomic(float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load float** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast float* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast float* [[TMP0]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 4, i8* [[T1]], i8* [[T2]], i32 5)
// CHECK-NEXT: [[T3:%.*]] = load float* [[TMP0]], align 4
// CHECK-NEXT: store float [[T3]], float* [[F]]
  float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load float* [[F]], align 4
// CHECK-NEXT: [[T1:%.*]] = load float** [[FP]], align 4
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

// CHECK-NEXT: [[P:%.*]] = load [[CF]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[P]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]]* [[P]], i32 0, i32 1
// CHECK-NEXT: store float 1.000000e+00, float* [[T0]]
// CHECK-NEXT: store float 0.000000e+00, float* [[T1]]
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[X]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]]* [[X]], i32 0, i32 1
// CHECK-NEXT: store float 2.000000e+00, float* [[T0]]
// CHECK-NEXT: store float 0.000000e+00, float* [[T1]]
  _Atomic(_Complex float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load [[CF]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[CF]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[CF]]* [[TMP0]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 8, i8* [[T1]], i8* [[T2]], i32 5)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[TMP0]], i32 0, i32 0
// CHECK-NEXT: [[R:%.*]] = load float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[TMP0]], i32 0, i32 1
// CHECK-NEXT: [[I:%.*]] = load float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[F]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]]* [[F]], i32 0, i32 1
// CHECK-NEXT: store float [[R]], float* [[T0]]
// CHECK-NEXT: store float [[I]], float* [[T1]]
  _Complex float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[F]], i32 0, i32 0
// CHECK-NEXT: [[R:%.*]] = load float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[F]], i32 0, i32 1
// CHECK-NEXT: [[I:%.*]] = load float* [[T0]]
// CHECK-NEXT: [[DEST:%.*]] = load [[CF]]** [[FP]], align 4
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[CF]]* [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[CF]]* [[TMP1]], i32 0, i32 1
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

// CHECK-NEXT: [[P:%.*]] = load [[S]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[P]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[P]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[P]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[P]], i32 0, i32 3
// CHECK-NEXT: store i16 4, i16* [[T0]], align 2
  __c11_atomic_init(fp, (S){1,2,3,4});

// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[X]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[X]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[X]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T0]], align 2
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[S]]* [[X]], i32 0, i32 3
// CHECK-NEXT: store i16 4, i16* [[T0]], align 2
  _Atomic(S) x = (S){1,2,3,4};

// CHECK-NEXT: [[T0:%.*]] = load [[S]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[S]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[S]]* [[F]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 8, i8* [[T1]], i8* [[T2]], i32 5)
  S f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load [[S]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[S]]* [[TMP0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[S]]* [[F]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[T1]], i8* [[T2]], i32 8, i32 2, i1 false)
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

// CHECK-NEXT: [[P:%.*]] = load [[APS]]** [[FP]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[APS]]* [[P]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 8, i32 8, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]]* [[P]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T1]], align 2
  __c11_atomic_init(fp, (PS){1,2,3});

// CHECK-NEXT: [[T0:%.*]] = bitcast [[APS]]* [[X]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* [[T0]], i8 0, i32 8, i32 8, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]]* [[X]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 0
// CHECK-NEXT: store i16 1, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 1
// CHECK-NEXT: store i16 2, i16* [[T1]], align 2
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[PS]]* [[T0]], i32 0, i32 2
// CHECK-NEXT: store i16 3, i16* [[T1]], align 2
  _Atomic(PS) x = (PS){1,2,3};

// CHECK-NEXT: [[T0:%.*]] = load [[APS]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[APS]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[APS]]* [[TMP0]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 8, i8* [[T1]], i8* [[T2]], i32 5)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]]* [[TMP0]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = bitcast [[PS]]* [[F]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[PS]]* [[T0]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[T1]], i8* [[T2]], i32 6, i32 2, i1 false)
  PS f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load [[APS]]** [[FP]]
// CHECK-NEXT: [[T1:%.*]] = bitcast { %struct.PS, [2 x i8] }* [[TMP1]] to i8*
// CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* [[T1]], i8 0, i32 8, i32 8, i1 false)
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[APS]]* [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[T2:%.*]] = bitcast [[PS]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = bitcast [[PS]]* [[F]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[T2]], i8* [[T3]], i32 6, i32 2, i1 false)
// CHECK-NEXT: [[T4:%.*]] = bitcast [[APS]]* [[T0]] to i8*
// CHECK-NEXT: [[T5:%.*]] = bitcast [[APS]]* [[TMP1]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 8, i8* [[T4]], i8* [[T5]], i32 5)
  *fp = f;

// CHECK-NEXT: [[T0:%.*]] = load [[APS]]** [[FP]], align 4
// CHECK-NEXT: [[T1:%.*]] = bitcast [[APS]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[APS]]* [[TMP3]] to i8*
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 8, i8* [[T1]], i8* [[T2]], i32 5)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]]* [[TMP3]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = bitcast %struct.PS* [[TMP2]] to i8*
// CHECK-NEXT: [[T2:%.*]] = bitcast %struct.PS* [[T0]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[T1]], i8* [[T2]], i32 6, i32 2, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds %struct.PS* [[TMP2]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = load i16* [[T0]], align 2
// CHECK-NEXT: [[T2:%.*]] = sext i16 [[T1]] to i32
// CHECK-NEXT: store i32 [[T2]], i32* [[A]], align 4
  int a = ((PS)*fp).x;

// CHECK-NEXT: ret void
}

// CHECK: define arm_aapcscc void @testPromotedStructOps([[APS:.*]]*

// FIXME: none of these look right, but we can leave the "test" here
// to make sure they at least don't crash.
void testPromotedStructOps(_Atomic(PS) *p) {
  PS a = __c11_atomic_load(p, 5);
  __c11_atomic_store(p, a, 5);
  PS b = __c11_atomic_exchange(p, a, 5);
  _Bool v = __c11_atomic_compare_exchange_strong(p, &b, a, 5, 5);
  v = __c11_atomic_compare_exchange_weak(p, &b, a, 5, 5);
}
