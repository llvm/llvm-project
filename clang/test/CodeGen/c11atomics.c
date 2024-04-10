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
// CHECK-DAG: %struct.ptr = type { ptr }

struct elem {
    _Atomic(struct ptr) link;
};

struct ptr object;
// CHECK-DAG: @object ={{.*}} global %struct.ptr zeroinitializer

// CHECK-DAG: @testStructGlobal ={{.*}} global {{.*}} { i16 1, i16 2, i16 3, i16 4 }
// CHECK-DAG: @testPromotedStructGlobal ={{.*}} global {{.*}} { %{{.*}} { i16 1, i16 2, i16 3 }, [2 x i8] zeroinitializer }


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
  // CHECK: atomicrmw xchg ptr @b, i8 1 seq_cst, align 1
  b++;
  // CHECK: atomicrmw add ptr @i, i32 1 seq_cst, align 4
  i++;
  // CHECK: atomicrmw add ptr @l, i64 1 seq_cst, align 8
  l++;
  // CHECK: atomicrmw add ptr @s, i16 1 seq_cst, align 2
  s++;
  // Prefix increment
  // Special case for bool: set to true and return true
  // CHECK: store atomic i8 1, ptr @b seq_cst, align 1
  ++b;
  // Currently, we have no variant of atomicrmw that returns the new value, so
  // we have to generate an atomic add, which returns the old value, and then a
  // non-atomic add.
  // CHECK: atomicrmw add ptr @i, i32 1 seq_cst, align 4
  // CHECK: add i32
  ++i;
  // CHECK: atomicrmw add ptr @l, i64 1 seq_cst, align 8
  // CHECK: add i64
  ++l;
  // CHECK: atomicrmw add ptr @s, i16 1 seq_cst, align 2
  // CHECK: add i16
  ++s;
}
// CHECK: testdec
void testdec(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 noundef 1, ptr noundef @b
  b--;
  // CHECK: atomicrmw sub ptr @i, i32 1 seq_cst, align 4
  i--;
  // CHECK: atomicrmw sub ptr @l, i64 1 seq_cst, align 8
  l--;
  // CHECK: atomicrmw sub ptr @s, i16 1 seq_cst, align 2
  s--;
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 noundef 1, ptr noundef @b
  --b;
  // CHECK: atomicrmw sub ptr @i, i32 1 seq_cst, align 4
  // CHECK: sub i32
  --i;
  // CHECK: atomicrmw sub ptr @l, i64 1 seq_cst, align 8
  // CHECK: sub i64
  --l;
  // CHECK: atomicrmw sub ptr @s, i16 1 seq_cst, align 2
  // CHECK: sub i16
  --s;
}
// CHECK: testaddeq
void testaddeq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 noundef 1, ptr noundef @b
  // CHECK: atomicrmw add ptr @i, i32 42 seq_cst, align 4
  // CHECK: atomicrmw add ptr @l, i64 42 seq_cst, align 8
  // CHECK: atomicrmw add ptr @s, i16 42 seq_cst, align 2
  b += 42;
  i += 42;
  l += 42;
  s += 42;
}
// CHECK: testsubeq
void testsubeq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 noundef 1, ptr noundef @b
  // CHECK: atomicrmw sub ptr @i, i32 42 seq_cst, align 4
  // CHECK: atomicrmw sub ptr @l, i64 42 seq_cst, align 8
  // CHECK: atomicrmw sub ptr @s, i16 42 seq_cst, align 2
  b -= 42;
  i -= 42;
  l -= 42;
  s -= 42;
}
// CHECK: testxoreq
void testxoreq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 noundef 1, ptr noundef @b
  // CHECK: atomicrmw xor ptr @i, i32 42 seq_cst, align 4
  // CHECK: atomicrmw xor ptr @l, i64 42 seq_cst, align 8
  // CHECK: atomicrmw xor ptr @s, i16 42 seq_cst, align 2
  b ^= 42;
  i ^= 42;
  l ^= 42;
  s ^= 42;
}
// CHECK: testoreq
void testoreq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 noundef 1, ptr noundef @b
  // CHECK: atomicrmw or ptr @i, i32 42 seq_cst, align 4
  // CHECK: atomicrmw or ptr @l, i64 42 seq_cst, align 8
  // CHECK: atomicrmw or ptr @s, i16 42 seq_cst, align 2
  b |= 42;
  i |= 42;
  l |= 42;
  s |= 42;
}
// CHECK: testandeq
void testandeq(void)
{
  // CHECK: call arm_aapcscc zeroext i1 @__atomic_compare_exchange(i32 noundef 1, ptr noundef @b
  // CHECK: atomicrmw and ptr @i, i32 42 seq_cst, align 4
  // CHECK: atomicrmw and ptr @l, i64 42 seq_cst, align 8
  // CHECK: atomicrmw and ptr @s, i16 42 seq_cst, align 2
  b &= 42;
  i &= 42;
  l &= 42;
  s &= 42;
}

// CHECK-LABEL: define{{.*}} arm_aapcscc void @testFloat(ptr
void testFloat(_Atomic(float) *fp) {
// CHECK:      [[FP:%.*]] = alloca ptr
// CHECK-NEXT: [[X:%.*]] = alloca float
// CHECK-NEXT: [[F:%.*]] = alloca float
// CHECK-NEXT: [[TMP0:%.*]] = alloca float
// CHECK-NEXT: [[TMP1:%.*]] = alloca float
// CHECK-NEXT: store ptr {{%.*}}, ptr [[FP]]

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: store float 1.000000e+00, ptr [[T0]], align 4
  __c11_atomic_init(fp, 1.0f);

// CHECK-NEXT: store float 2.000000e+00, ptr [[X]], align 4
  _Atomic(float) x = 2.0f;

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 noundef 4, ptr noundef [[T0]], ptr noundef [[TMP0]], i32 noundef 5)
// CHECK-NEXT: [[T3:%.*]] = load float, ptr [[TMP0]], align 4
// CHECK-NEXT: store float [[T3]], ptr [[F]]
  float f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load float, ptr [[F]], align 4
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[FP]], align 4
// CHECK-NEXT: store float [[T0]], ptr [[TMP1]], align 4
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 noundef 4, ptr noundef [[T1]], ptr noundef [[TMP1]], i32 noundef 5)
  *fp = f;

// CHECK-NEXT: ret void
}

// CHECK: define{{.*}} arm_aapcscc void @testComplexFloat(ptr
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
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 noundef 8, ptr noundef [[T0]], ptr noundef [[TMP0]], i32 noundef 5)
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
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 noundef 8, ptr noundef [[DEST]], ptr noundef [[TMP1]], i32 noundef 5)
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z, w; } S;
_Atomic S testStructGlobal = (S){1, 2, 3, 4};
// CHECK: define{{.*}} arm_aapcscc void @testStruct(ptr
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
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 noundef 8, ptr noundef [[T0]], ptr noundef [[F]], i32 noundef 5)
  S f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[TMP0]], ptr align 2 [[F]], i32 8, i1 false)
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 noundef 8, ptr noundef [[T0]], ptr noundef [[TMP0]], i32 noundef 5)
  *fp = f;

// CHECK-NEXT: ret void
}

typedef struct { short x, y, z; } PS;
_Atomic PS testPromotedStructGlobal = (PS){1, 2, 3};
// CHECK: define{{.*}} arm_aapcscc void @testPromotedStruct(ptr
void testPromotedStruct(_Atomic(PS) *fp) {
// CHECK:      [[FP:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[X:%.*]] = alloca [[APS:.*]], align 8
// CHECK-NEXT: [[F:%.*]] = alloca [[PS:%.*]], align 2
// CHECK-NEXT: [[TMP0:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[TMP1:%.*]] = alloca [[APS]], align 8
// CHECK-NEXT: [[A:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[TMP2:%.*]] = alloca %struct.PS, align 2
// CHECK-NEXT: [[TMP3:%.*]] = alloca [[APS]], align 8
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
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 noundef 8, ptr noundef [[T0]], ptr noundef [[TMP0]], i32 noundef 5)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], ptr [[TMP0]], i32 0, i32 0
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[F]], ptr align 8 [[T0]], i32 6, i1 false)
  PS f = *fp;

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]]
// CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 8 [[TMP1]], i8 0, i32 8, i1 false)
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[APS]], ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[T1]], ptr align 2 [[F]], i32 6, i1 false)
// CHECK-NEXT: call arm_aapcscc void @__atomic_store(i32 noundef 8, ptr noundef [[T0]], ptr noundef [[TMP1]], i32 noundef 5)
  *fp = f;

// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FP]], align 4
// CHECK-NEXT: call arm_aapcscc void @__atomic_load(i32 noundef 8, ptr noundef [[T0]], ptr noundef [[TMP3]], i32 noundef 5)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[APS]], ptr [[TMP3]], i32 0, i32 0
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[TMP2]], ptr align 8 [[T0]], i32 6, i1 false)
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds %struct.PS, ptr [[TMP2]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = load i16, ptr [[T0]], align 2
// CHECK-NEXT: [[T2:%.*]] = sext i16 [[T1]] to i32
// CHECK-NEXT: store i32 [[T2]], ptr [[A]], align 4
  int a = ((PS)*fp).x;

// CHECK-NEXT: ret void
}

PS test_promoted_load(_Atomic(PS) *addr) {
  // CHECK-LABEL: @test_promoted_load(ptr dead_on_unwind noalias writable sret(%struct.PS) align 2 %agg.result, ptr noundef %addr)
  // CHECK:   [[ADDR_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[ATOMIC_RES:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store ptr %addr, ptr [[ADDR_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load ptr, ptr [[ADDR_ARG]], align 4
  // CHECK:   [[ATOMIC_RES:%.*]] = load atomic i64, ptr [[ADDR]] seq_cst, align 8
  // CHECK:   store i64 [[ATOMIC_RES]], ptr [[ATOMIC_RES_ADDR:%.*]], align 8
  // CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 2 %agg.result, ptr align 8 [[ATOMIC_RES_ADDR]], i32 6, i1 false)
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
  // CHECK:   [[ATOMIC:%.*]] = load i64, ptr [[ATOMIC_VAL]], align 8
  // CHECK:   store atomic i64 [[ATOMIC]], ptr [[ADDR]] seq_cst, align 8
  __c11_atomic_store(addr, *val, 5);
}

PS test_promoted_exchange(_Atomic(PS) *addr, PS *val) {
  // CHECK-LABEL: @test_promoted_exchange(ptr dead_on_unwind noalias writable sret(%struct.PS) align 2 %agg.result, ptr noundef %addr, ptr noundef %val)
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
  // CHECK:   [[ATOMIC:%.*]] = load i64, ptr [[ATOMIC_VAL]], align 8
  // CHECK:   [[ATOMIC_RES:%.*]] = atomicrmw xchg ptr [[ADDR]], i64 [[ATOMIC]] seq_cst, align 8
  // CHECK:   store i64 [[ATOMIC_RES]], ptr [[ATOMIC_RES_PTR:%.*]], align 8
  // CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 2 %agg.result, ptr align 8 [[ATOMIC_RES_PTR]], i32 6, i1 false)
  return __c11_atomic_exchange(addr, *val, 5);
}

_Bool test_promoted_cmpxchg(_Atomic(PS) *addr, PS *desired, PS *new) {
  // CHECK-LABEL: i1 @test_promoted_cmpxchg(ptr noundef %addr, ptr noundef %desired, ptr noundef %new) #0 {
  // CHECK:   [[ADDR_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[DESIRED_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[NEW_ARG:%.*]] = alloca ptr, align 4
  // CHECK:   [[NONATOMIC_TMP:%.*]] = alloca %struct.PS, align 2
  // CHECK:   [[ATOMIC_DESIRED:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   [[ATOMIC_NEW:%.*]] = alloca { %struct.PS, [2 x i8] }, align 8
  // CHECK:   store ptr %addr, ptr [[ADDR_ARG]], align 4
  // CHECK:   store ptr %desired, ptr [[DESIRED_ARG]], align 4
  // CHECK:   store ptr %new, ptr [[NEW_ARG]], align 4
  // CHECK:   [[ADDR:%.*]] = load ptr, ptr [[ADDR_ARG]], align 4
  // CHECK:   [[DESIRED:%.*]] = load ptr, ptr [[DESIRED_ARG]], align 4
  // CHECK:   [[NEW:%.*]] = load ptr, ptr [[NEW_ARG]], align 4
  // CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[NONATOMIC_TMP]], ptr align 2 [[NEW]], i32 6, i1 false)
  // CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ATOMIC_DESIRED]], ptr align 2 [[DESIRED]], i64 6, i1 false)
  // CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ATOMIC_NEW]], ptr align 2 [[NONATOMIC_TMP]], i64 6, i1 false)
  // CHECK:   [[VAL1:%.*]] = load i64, ptr [[ATOMIC_DESIRED]], align 8
  // CHECK:   [[VAL2:%.*]] = load i64, ptr [[ATOMIC_NEW]], align 8
  // CHECK:   [[RES_PAIR:%.*]] = cmpxchg ptr [[ADDR]], i64 [[VAL1]], i64 [[VAL2]] seq_cst seq_cst, align 8
  // CHECK:   [[RES:%.*]] = extractvalue { i64, i1 } [[RES_PAIR]], 1
  return __c11_atomic_compare_exchange_strong(addr, desired, *new, 5, 5);
}

struct Empty {};

struct Empty test_empty_struct_load(_Atomic(struct Empty)* empty) {
  // CHECK-LABEL: @test_empty_struct_load(
  // CHECK: load atomic i8, ptr {{.*}}, align 1
  return __c11_atomic_load(empty, 5);
}

void test_empty_struct_store(_Atomic(struct Empty)* empty, struct Empty value) {
  // CHECK-LABEL: @test_empty_struct_store(
  // CHECK: store atomic i8 {{.*}}, ptr {{.*}}, align 1
  __c11_atomic_store(empty, value, 5);
}
