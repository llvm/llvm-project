// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Available on resource dir.
#include <stdatomic.h>

typedef struct _a {
  _Atomic(int) d;
} at;

void m() { at y; }

signed char sc;
unsigned char uc;
signed short ss;
unsigned short us;
signed int si;
unsigned int ui;
signed long long sll;
unsigned long long ull;

// CHECK: ![[A:.*]] = !cir.struct<struct "_a" {!s32i}>

int basic_binop_fetch(int *i) {
  return __atomic_add_fetch(i, 1, memory_order_seq_cst);
}

// CHECK: cir.func @_Z17basic_binop_fetchPi
// CHECK:  %[[ARGI:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["i", init] {alignment = 8 : i64}
// CHECK:  %[[ONE_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, [".atomictmp"] {alignment = 4 : i64}
// CHECK:  cir.store %arg0, %[[ARGI]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK:  %[[I:.*]] = cir.load %[[ARGI]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:  %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:  cir.store %[[ONE]], %[[ONE_ADDR]] : !s32i, !cir.ptr<!s32i>
// CHECK:  %[[VAL:.*]] = cir.load %[[ONE_ADDR]] : !cir.ptr<!s32i>, !s32i
// CHECK:  cir.atomic.fetch(add, %[[I]] : !cir.ptr<!s32i>, %[[VAL]] : !s32i, seq_cst) : !s32i

// LLVM: define dso_local i32 @_Z17basic_binop_fetchPi
// LLVM: %[[RMW:.*]] = atomicrmw add ptr {{.*}}, i32 %[[VAL:.*]] seq_cst, align 4
// LLVM: add i32 %[[RMW]], %[[VAL]]

int other_binop_fetch(int *i) {
  __atomic_sub_fetch(i, 1, memory_order_relaxed);
  __atomic_and_fetch(i, 1, memory_order_consume);
  __atomic_or_fetch(i, 1, memory_order_acquire);
  return __atomic_xor_fetch(i, 1, memory_order_release);
}

// CHECK: cir.func @_Z17other_binop_fetchPi
// CHECK: cir.atomic.fetch(sub, {{.*}}, relaxed
// CHECK: cir.atomic.fetch(and, {{.*}}, acquire
// CHECK: cir.atomic.fetch(or, {{.*}}, acquire
// CHECK: cir.atomic.fetch(xor, {{.*}}, release

// LLVM: define dso_local i32 @_Z17other_binop_fetchPi
// LLVM: %[[RMW_SUB:.*]] = atomicrmw sub ptr {{.*}} monotonic
// LLVM: sub i32 %[[RMW_SUB]], {{.*}}
// LLVM: %[[RMW_AND:.*]] = atomicrmw and ptr {{.*}} acquire
// LLVM: and i32 %[[RMW_AND]], {{.*}}
// LLVM: %[[RMW_OR:.*]] = atomicrmw or ptr {{.*}} acquire
// LLVM: or i32 %[[RMW_OR]], {{.*}}
// LLVM: %[[RMW_XOR:.*]] = atomicrmw xor ptr {{.*}} release
// LLVM: xor i32 %[[RMW_XOR]], {{.*}}

int nand_binop_fetch(int *i) {
  return __atomic_nand_fetch(i, 1, memory_order_acq_rel);
}

// CHECK: cir.func @_Z16nand_binop_fetchPi
// CHECK: cir.atomic.fetch(nand, {{.*}}, acq_rel

// LLVM: define dso_local i32 @_Z16nand_binop_fetchPi
// LLVM: %[[RMW_NAND:.*]] = atomicrmw nand ptr {{.*}} acq_rel
// LLVM: %[[AND:.*]] = and i32 %[[RMW_NAND]]
// LLVM: = xor i32 %[[AND]], -1

int fp_binop_fetch(float *i) {
  __atomic_add_fetch(i, 1, memory_order_seq_cst);
  return __atomic_sub_fetch(i, 1, memory_order_seq_cst);
}

// CHECK: cir.func @_Z14fp_binop_fetchPf
// CHECK: cir.atomic.fetch(add,
// CHECK: cir.atomic.fetch(sub,

// LLVM: define dso_local i32 @_Z14fp_binop_fetchPf
// LLVM: %[[RMW_FADD:.*]] = atomicrmw fadd ptr
// LLVM: fadd float %[[RMW_FADD]]
// LLVM: %[[RMW_FSUB:.*]] = atomicrmw fsub ptr
// LLVM: fsub float %[[RMW_FSUB]]

int fetch_binop(int *i) {
  __atomic_fetch_add(i, 1, memory_order_seq_cst);
  __atomic_fetch_sub(i, 1, memory_order_seq_cst);
  __atomic_fetch_and(i, 1, memory_order_seq_cst);
  __atomic_fetch_or(i, 1, memory_order_seq_cst);
  __atomic_fetch_xor(i, 1, memory_order_seq_cst);
  return __atomic_fetch_nand(i, 1, memory_order_seq_cst);
}

// CHECK: cir.func @_Z11fetch_binopPi
// CHECK: cir.atomic.fetch(add, {{.*}}) fetch_first
// CHECK: cir.atomic.fetch(sub, {{.*}}) fetch_first
// CHECK: cir.atomic.fetch(and, {{.*}}) fetch_first
// CHECK: cir.atomic.fetch(or, {{.*}}) fetch_first
// CHECK: cir.atomic.fetch(xor, {{.*}}) fetch_first
// CHECK: cir.atomic.fetch(nand, {{.*}}) fetch_first

// LLVM: define dso_local i32 @_Z11fetch_binopPi
// LLVM: atomicrmw add ptr
// LLVM-NOT: add {{.*}}
// LLVM: atomicrmw sub ptr
// LLVM-NOT: sub {{.*}}
// LLVM: atomicrmw and ptr
// LLVM-NOT: and {{.*}}
// LLVM: atomicrmw or ptr
// LLVM-NOT: or {{.*}}
// LLVM: atomicrmw xor ptr
// LLVM-NOT: xor {{.*}}
// LLVM: atomicrmw nand ptr
// LLVM-NOT: nand {{.*}}

void min_max_fetch(int *i) {
  __atomic_fetch_max(i, 1, memory_order_seq_cst);
  __atomic_fetch_min(i, 1, memory_order_seq_cst);
  __atomic_max_fetch(i, 1, memory_order_seq_cst);
  __atomic_min_fetch(i, 1, memory_order_seq_cst);
}

// CHECK: cir.func @_Z13min_max_fetchPi
// CHECK: = cir.atomic.fetch(max, {{.*}}) fetch_first
// CHECK: = cir.atomic.fetch(min, {{.*}}) fetch_first
// CHECK: = cir.atomic.fetch(max, {{.*}}) : !s32i
// CHECK: = cir.atomic.fetch(min, {{.*}}) : !s32i

// LLVM: define dso_local void @_Z13min_max_fetchPi
// LLVM: atomicrmw max ptr
// LLVM-NOT: icmp {{.*}}
// LLVM: atomicrmw min ptr
// LLVM-NOT: icmp {{.*}}
// LLVM: %[[MAX:.*]] = atomicrmw max ptr
// LLVM: %[[ICMP_MAX:.*]] = icmp sgt i32 %[[MAX]]
// LLVM: select i1 %[[ICMP_MAX]], i32 %[[MAX]]
// LLVM: %[[MIN:.*]] = atomicrmw min ptr
// LLVM: %[[ICMP_MIN:.*]] = icmp slt i32 %[[MIN]]
// LLVM: select i1 %[[ICMP_MIN]], i32 %[[MIN]]

int fi1(_Atomic(int) *i) {
  return __c11_atomic_load(i, memory_order_seq_cst);
}

// CHECK: cir.func @_Z3fi1PU7_Atomici
// CHECK: cir.load atomic(seq_cst)

// LLVM-LABEL: @_Z3fi1PU7_Atomici
// LLVM: load atomic i32, ptr {{.*}} seq_cst, align 4

int fi1a(int *i) {
  int v;
  __atomic_load(i, &v, memory_order_seq_cst);
  return v;
}

// CHECK-LABEL: @_Z4fi1aPi
// CHECK: cir.load atomic(seq_cst)

// LLVM-LABEL: @_Z4fi1aPi
// LLVM: load atomic i32, ptr {{.*}} seq_cst, align 4

int fi1b(int *i) {
  return __atomic_load_n(i, memory_order_seq_cst);
}

// CHECK-LABEL: @_Z4fi1bPi
// CHECK: cir.load atomic(seq_cst)

// LLVM-LABEL: @_Z4fi1bPi
// LLVM: load atomic i32, ptr {{.*}} seq_cst, align 4

int fi1c(atomic_int *i) {
  return atomic_load(i);
}

// CHECK-LABEL: @_Z4fi1cPU7_Atomici
// CHECK: cir.load atomic(seq_cst)

// LLVM-LABEL: @_Z4fi1cPU7_Atomici
// LLVM: load atomic i32, ptr {{.*}} seq_cst, align 4

void fi2(_Atomic(int) *i) {
  __c11_atomic_store(i, 1, memory_order_seq_cst);
}

// CHECK-LABEL: @_Z3fi2PU7_Atomici
// CHECK: cir.store atomic(seq_cst)

// LLVM-LABEL: @_Z3fi2PU7_Atomici
// LLVM: store atomic i32 {{.*}} seq_cst, align 4

void fi2a(int *i) {
  int v = 1;
  __atomic_store(i, &v, memory_order_seq_cst);
}

// CHECK-LABEL: @_Z4fi2aPi
// CHECK: cir.store atomic(seq_cst)

// LLVM-LABEL: @_Z4fi2aPi
// LLVM: store atomic i32 {{.*}} seq_cst, align 4

void fi2b(int *i) {
  __atomic_store_n(i, 1, memory_order_seq_cst);
}

// CHECK-LABEL: @_Z4fi2bPi
// CHECK: cir.store atomic(seq_cst)

// LLVM-LABEL: @_Z4fi2bPi
// LLVM: store atomic i32 {{.*}} seq_cst, align 4

void fi2c(atomic_int *i) {
  atomic_store(i, 1);
}

struct S {
  double x;
};

// CHECK-LABEL: @_Z4fi2cPU7_Atomici
// CHECK: cir.store atomic(seq_cst)

// LLVM-LABEL: @_Z4fi2cPU7_Atomici
// LLVM: store atomic i32 {{.*}} seq_cst, align 4

void fd3(struct S *a, struct S *b, struct S *c) {
  __atomic_exchange(a, b, c, memory_order_seq_cst);
}

// CHECK-LABEL: @_Z3fd3P1SS0_S0_
// CHECK: cir.atomic.xchg({{.*}} : !cir.ptr<!u64i>, {{.*}} : !u64i, seq_cst) : !u64i

// FIXME: CIR is producing an over alignment of 8, only 4 needed.
// LLVM-LABEL: @_Z3fd3P1SS0_S0_
// LLVM:      [[A_ADDR:%.*]] = alloca ptr
// LLVM-NEXT: [[B_ADDR:%.*]] = alloca ptr
// LLVM-NEXT: [[C_ADDR:%.*]] = alloca ptr
// LLVM-NEXT: store ptr {{.*}}, ptr [[A_ADDR]]
// LLVM-NEXT: store ptr {{.*}}, ptr [[B_ADDR]]
// LLVM-NEXT: store ptr {{.*}}, ptr [[C_ADDR]]
// LLVM-NEXT: [[LOAD_A_PTR:%.*]] = load ptr, ptr [[A_ADDR]]
// LLVM-NEXT: [[LOAD_B_PTR:%.*]] = load ptr, ptr [[B_ADDR]]
// LLVM-NEXT: [[LOAD_C_PTR:%.*]] = load ptr, ptr [[C_ADDR]]
// LLVM-NEXT: [[LOAD_B:%.*]] = load i64, ptr [[LOAD_B_PTR]]
// LLVM-NEXT: [[RESULT:%.*]] = atomicrmw xchg ptr [[LOAD_A_PTR]], i64 [[LOAD_B]] seq_cst
// LLVM-NEXT: store i64 [[RESULT]], ptr [[LOAD_C_PTR]]

bool fd4(struct S *a, struct S *b, struct S *c) {
  return __atomic_compare_exchange(a, b, c, 1, 5, 5);
}

// CHECK-LABEL: @_Z3fd4P1SS0_S0_
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!u64i>, {{.*}} : !u64i, {{.*}} : !u64i, success = seq_cst, failure = seq_cst) syncscope(system) align(8) weak : (!u64i, !cir.bool)

// LLVM-LABEL: @_Z3fd4P1SS0_S0_
// LLVM: cmpxchg weak ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8

bool fi4a(int *i) {
  int cmp = 0;
  int desired = 1;
  return __atomic_compare_exchange(i, &cmp, &desired, false, memory_order_acquire, memory_order_acquire);
}

// CHECK-LABEL: @_Z4fi4aPi
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s32i>, {{.*}} : !s32i, {{.*}} : !s32i, success = acquire, failure = acquire) syncscope(system) align(4) : (!s32i, !cir.bool)

// LLVM-LABEL: @_Z4fi4aPi
// LLVM: %[[RES:.*]] = cmpxchg ptr %7, i32 %8, i32 %9 acquire acquire, align 4
// LLVM: extractvalue { i32, i1 } %[[RES]], 0
// LLVM: extractvalue { i32, i1 } %[[RES]], 1

bool fi4b(int *i) {
  int cmp = 0;
  return __atomic_compare_exchange_n(i, &cmp, 1, true, memory_order_acquire, memory_order_acquire);
}

// CHECK-LABEL: @_Z4fi4bPi
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s32i>, {{.*}} : !s32i, {{.*}} : !s32i, success = acquire, failure = acquire) syncscope(system) align(4) weak : (!s32i, !cir.bool)

// LLVM-LABEL: @_Z4fi4bPi
// LLVM: %[[R:.*]] = cmpxchg weak ptr {{.*}}, i32 {{.*}}, i32 {{.*}} acquire acquire, align 4
// LLVM: extractvalue { i32, i1 } %[[R]], 0
// LLVM: extractvalue { i32, i1 } %[[R]], 1

bool fi4c(atomic_int *i) {
  int cmp = 0;
  return atomic_compare_exchange_strong(i, &cmp, 1);
}

// CHECK-LABEL: @_Z4fi4cPU7_Atomici
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s32i>, {{.*}} : !s32i, {{.*}} : !s32i, success = seq_cst, failure = seq_cst) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK: %[[CMP:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK: cir.if %[[CMP:.*]] {
// CHECK:   cir.store %old, {{.*}} : !s32i, !cir.ptr<!s32i>
// CHECK: }

// LLVM-LABEL: @_Z4fi4cPU7_Atomici
// LLVM: cmpxchg ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4

bool fi4d(atomic_int *i) {
  int cmp = 0;
  return atomic_compare_exchange_weak(i, &cmp, 1);
}

// CHECK-LABEL: @_Z4fi4dPU7_Atomici
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s32i>, {{.*}} : !s32i, {{.*}} : !s32i, success = seq_cst, failure = seq_cst) syncscope(system) align(4) weak : (!s32i, !cir.bool)
// CHECK: %[[CMP:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK: cir.if %[[CMP:.*]] {
// CHECK:   cir.store %old, {{.*}} : !s32i, !cir.ptr<!s32i>
// CHECK: }

// LLVM-LABEL: @_Z4fi4dPU7_Atomici
// LLVM: cmpxchg weak ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4

bool fsb(bool *c) {
  return __atomic_exchange_n(c, 1, memory_order_seq_cst);
}

// CHECK-LABEL: @_Z3fsbPb
// CHECK: cir.atomic.xchg({{.*}} : !cir.ptr<!u8i>, {{.*}} : !u8i, seq_cst) : !u8i

// LLVM-LABEL: @_Z3fsbPb
// LLVM: atomicrmw xchg ptr {{.*}}, i8 {{.*}} seq_cst, align 1

void atomicinit(void)
{
  _Atomic(unsigned int) j = 12;
  __c11_atomic_init(&j, 1);
}

// CHECK-LABEL: @_Z10atomicinitv
// CHECK: %[[ADDR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["j"
// CHECK: cir.store {{.*}}, %[[ADDR]] : !u32i, !cir.ptr<!u32i>
// CHECK: cir.store {{.*}}, %[[ADDR]] : !u32i, !cir.ptr<!u32i>

// LLVM-LABEL: @_Z10atomicinitv
// LLVM: %[[ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 12, ptr %[[ADDR]], align 4
// LLVM: store i32 1, ptr %[[ADDR]], align 4

void incdec() {
  _Atomic(unsigned int) j = 12;
  __c11_atomic_fetch_add(&j, 1, 0);
  __c11_atomic_fetch_sub(&j, 1, 0);
}

// CHECK-LABEL: @_Z6incdecv
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!u32i>, {{.*}} : !u32i, relaxed) fetch_first
// CHECK: cir.atomic.fetch(sub, {{.*}} : !cir.ptr<!u32i>, {{.*}} : !u32i, relaxed) fetch_first

// LLVM-LABEL: @_Z6incdecv
// LLVM: atomicrmw add ptr {{.*}}, i32 {{.*}} monotonic, align 4
// LLVM: atomicrmw sub ptr {{.*}}, i32 {{.*}} monotonic, align 4

void inc_int(int* a, int b) {
  int c = __sync_fetch_and_add(a, b);
}
// CHECK-LABEL: @_Z7inc_int
// CHECK: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[VAL:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[RES:.*]] = cir.atomic.fetch(add, %[[PTR]] : !cir.ptr<!s32i>, %[[VAL]] : !s32i, seq_cst) fetch_first : !s32i
// CHECK: cir.store %[[RES]], {{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: @_Z7inc_int
// LLVM: atomicrmw add ptr {{.*}}, i32 {{.*}} seq_cst, align 4

void sub_int(int* a, int b) {
  int c = __sync_fetch_and_sub(a, b);
}

// CHECK-LABEL: _Z7sub_int
// CHECK: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[VAL:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[RES:.*]] = cir.atomic.fetch(sub, %[[PTR]] : !cir.ptr<!s32i>, %[[VAL]] : !s32i, seq_cst) fetch_first : !s32i
// CHECK: cir.store %[[RES]], {{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: _Z7sub_int
// LLVM: atomicrmw sub ptr {{.*}}, i32 {{.*}} seq_cst, align 4


// CHECK-LABEL: @_Z8inc_long
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!s64i>, {{.*}} : !s64i, seq_cst) fetch_first : !s64i

// LLVM-LABEL: @_Z8inc_long
// LLVM: atomicrmw add ptr {{.*}}, i64 {{.*}} seq_cst, align 8

void inc_long(long* a, long b) {
  long c = __sync_fetch_and_add(a, 2);
}

// CHECK-LABEL: @_Z8sub_long
// CHECK: cir.atomic.fetch(sub, {{.*}} : !cir.ptr<!s64i>, {{.*}} : !s64i, seq_cst) fetch_first : !s64i

// LLVM-LABEL: @_Z8sub_long
// LLVM: atomicrmw sub ptr {{.*}}, i64 {{.*}} seq_cst, align 8

void sub_long(long* a, long b) {
  long c = __sync_fetch_and_sub(a, 2);
}


// CHECK-LABEL: @_Z9inc_short
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!s16i>, {{.*}} : !s16i, seq_cst) fetch_first : !s16i

// LLVM-LABEL: @_Z9inc_short
// LLVM: atomicrmw add ptr {{.*}}, i16 {{.*}} seq_cst, align 2
void inc_short(short* a, short b) {
  short c = __sync_fetch_and_add(a, 2);
}

// CHECK-LABEL: @_Z9sub_short
// CHECK: cir.atomic.fetch(sub, {{.*}} : !cir.ptr<!s16i>, {{.*}} : !s16i, seq_cst) fetch_first : !s16i

// LLVM-LABEL: @_Z9sub_short
// LLVM: atomicrmw sub ptr {{.*}}, i16 {{.*}} seq_cst, align 2
void sub_short(short* a, short b) {
  short c = __sync_fetch_and_sub(a, 2);
}


// CHECK-LABEL: @_Z8inc_byte
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!s8i>, {{.*}} : !s8i, seq_cst) fetch_first : !s8i

// LLVM-LABEL: @_Z8inc_byte
// LLVM: atomicrmw add ptr {{.*}}, i8 {{.*}} seq_cst, align 1
void inc_byte(char* a, char b) {
  char c = __sync_fetch_and_add(a, b);
}

// CHECK-LABEL: @_Z8sub_byte
// CHECK: cir.atomic.fetch(sub, {{.*}} : !cir.ptr<!s8i>, {{.*}} : !s8i, seq_cst) fetch_first : !s8i

// LLVM-LABEL: @_Z8sub_byte
// LLVM: atomicrmw sub ptr {{.*}}, i8 {{.*}} seq_cst, align 1
void sub_byte(char* a, char b) {
  char c = __sync_fetch_and_sub(a, b);
}

// CHECK-LABEL: @_Z12cmp_bool_int
// CHECK: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[CMP:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[UPD:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[OLD:.*]], %[[RES:.*]] = cir.atomic.cmp_xchg(%[[PTR]] : !cir.ptr<!s32i>, %[[CMP]] : !s32i, %[[UPD]] : !s32i, success = seq_cst, failure = seq_cst) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK: cir.store %[[RES]], {{.*}} : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: @_Z12cmp_bool_int
// LLVM: %[[PTR:.*]] = load ptr
// LLVM: %[[CMP:.*]] = load i32
// LLVM: %[[UPD:.*]] = load i32
// LLVM: %[[RES:.*]] = cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[UPD]] seq_cst seq_cst, align 4
// LLVM: %[[TMP:.*]] = extractvalue { i32, i1 } %[[RES]], 1
// LLVM: %[[EXT:.*]] = zext i1 %[[TMP]] to i8
// LLVM: store i8 %[[EXT]], ptr {{.*}}
void cmp_bool_int(int* p, int x, int u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}


// CHECK-LABEL: @_Z13cmp_bool_long
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s64i>, {{.*}} : !s64i, {{.*}} : !s64i, success = seq_cst, failure = seq_cst) syncscope(system) align(8) : (!s64i, !cir.bool)

// LLVM-LABEL: @_Z13cmp_bool_long
// LLVM: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8
void cmp_bool_long(long* p, long x, long u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z14cmp_bool_short
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s16i>, {{.*}} : !s16i, {{.*}} : !s16i, success = seq_cst, failure = seq_cst) syncscope(system) align(2) : (!s16i, !cir.bool)

// LLVM-LABEL: @_Z14cmp_bool_short
// LLVM: cmpxchg ptr {{.*}}, i16 {{.*}}, i16 {{.*}} seq_cst seq_cst, align 2
void cmp_bool_short(short* p, short x, short u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z13cmp_bool_byte
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s8i>, {{.*}} : !s8i, {{.*}} : !s8i, success = seq_cst, failure = seq_cst) syncscope(system) align(1) : (!s8i, !cir.bool)

// LLVM-LABEL: @_Z13cmp_bool_byte
// LLVM: cmpxchg ptr {{.*}}, i8 {{.*}}, i8 {{.*}} seq_cst seq_cst, align 1
void cmp_bool_byte(char* p, char x, char u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z11cmp_val_int
// CHECK: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[CMP:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[UPD:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[OLD:.*]], %[[RES:.*]] = cir.atomic.cmp_xchg(%[[PTR]] : !cir.ptr<!s32i>, %[[CMP]] : !s32i, %[[UPD]] : !s32i, success = seq_cst, failure = seq_cst) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK: cir.store %[[OLD]], {{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: @_Z11cmp_val_int
// LLVM: %[[PTR:.*]] = load ptr
// LLVM: %[[CMP:.*]] = load i32
// LLVM: %[[UPD:.*]] = load i32
// LLVM: %[[RES:.*]] = cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[UPD]] seq_cst seq_cst, align 4
// LLVM: %[[TMP:.*]] = extractvalue { i32, i1 } %[[RES]], 0
// LLVM: store i32 %[[TMP]], ptr {{.*}}
void cmp_val_int(int* p, int x, int u) {
  int r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z12cmp_val_long
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s64i>, {{.*}} : !s64i, {{.*}} : !s64i, success = seq_cst, failure = seq_cst) syncscope(system) align(8) : (!s64i, !cir.bool)

// LLVM-LABEL: @_Z12cmp_val_long
// LLVM: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8
void cmp_val_long(long* p, long x, long u) {
  long r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z13cmp_val_short
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s16i>, {{.*}} : !s16i, {{.*}} : !s16i, success = seq_cst, failure = seq_cst) syncscope(system) align(2) : (!s16i, !cir.bool)

// LLVM-LABEL: @_Z13cmp_val_short
// LLVM: cmpxchg ptr {{.*}}, i16 {{.*}}, i16 {{.*}} seq_cst seq_cst, align 2
void cmp_val_short(short* p, short x, short u) {
  short r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z12cmp_val_byte
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s8i>, {{.*}} : !s8i, {{.*}} : !s8i, success = seq_cst, failure = seq_cst) syncscope(system) align(1) : (!s8i, !cir.bool)

// LLVM-LABEL: @_Z12cmp_val_byte
// LLVM: cmpxchg ptr {{.*}}, i8 {{.*}}, i8 {{.*}} seq_cst seq_cst, align 1
void cmp_val_byte(char* p, char x, char u) {
  char r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z8inc_uint
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!u32i>, {{.*}} : !u32i, seq_cst) fetch_first : !u32i

// LLVM-LABEL: @_Z8inc_uint
// LLVM: atomicrmw add ptr {{.*}}, i32 {{.*}} seq_cst, align 4
void inc_uint(unsigned int* a, int b) {
  unsigned int c = __sync_fetch_and_add(a, b);
}

// CHECK-LABEL: @_Z8sub_uint
// CHECK: cir.atomic.fetch(sub, {{.*}} : !cir.ptr<!u32i>, {{.*}} : !u32i, seq_cst) fetch_first : !u32i

// LLVM-LABEL: @_Z8sub_uint
// LLVM: atomicrmw sub ptr {{.*}}, i32 {{.*}} seq_cst, align 4
void sub_uint(unsigned int* a, int b) {
  unsigned int c = __sync_fetch_and_sub(a, b);
}

// CHECK-LABEL: @_Z9inc_ulong
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!u64i>, {{.*}} : !u64i, seq_cst) fetch_first : !u64i

// LLVM-LABEL: @_Z9inc_ulong
// LLVM: atomicrmw add ptr {{.*}}, i64 {{.*}} seq_cst, align 8
void inc_ulong(unsigned long* a, long b) {
  unsigned long c = __sync_fetch_and_add(a, b);
}

// CHECK-LABEL: @_Z9sub_ulong
// CHECK: cir.atomic.fetch(sub, {{.*}} : !cir.ptr<!u64i>, {{.*}} : !u64i, seq_cst) fetch_first : !u64i

// LLVM-LABEL: @_Z9sub_ulong
// LLVM: atomicrmw sub ptr {{.*}}, i64 {{.*}} seq_cst, align 8
void sub_ulong(unsigned long* a, long b) {
  unsigned long c = __sync_fetch_and_sub(a, b);
}


// CHECK-LABEL: @_Z9inc_uchar
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!u8i>, {{.*}} : !u8i, seq_cst) fetch_first : !u8i

// LLVM-LABEL: @_Z9inc_uchar
// LLVM: atomicrmw add ptr {{.*}}, i8 {{.*}} seq_cst, align 1
void inc_uchar(unsigned char* a, char b) {
  unsigned char c = __sync_fetch_and_add(a, b);
}

// CHECK-LABEL: @_Z9sub_uchar
// CHECK: cir.atomic.fetch(sub, {{.*}} : !cir.ptr<!u8i>, {{.*}} : !u8i, seq_cst) fetch_first : !u8i

// LLVM-LABEL: @_Z9sub_uchar
// LLVM: atomicrmw sub ptr {{.*}}, i8 {{.*}} seq_cst, align 1
void sub_uchar(unsigned char* a, char b) {
  unsigned char c = __sync_fetch_and_sub(a, b);
}

// CHECK-LABEL: @_Z13cmp_bool_uint
// CHECK: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CHECK: %[[CMP:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[CMP_U:.*]] = cir.cast(integral, %[[CMP]] : !s32i), !u32i
// CHECK: %[[UPD:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[UPD_U:.*]] = cir.cast(integral, %[[UPD]] : !s32i), !u32i
// CHECK: %[[OLD:.*]], %[[RES:.*]] = cir.atomic.cmp_xchg(%[[PTR]] : !cir.ptr<!u32i>, %[[CMP_U]] :
// CHECK-SAME: !u32i, %[[UPD_U]] : !u32i, success = seq_cst, failure = seq_cst) syncscope(system) align(4) : (!u32i, !cir.bool)
// CHECK: cir.store %[[RES]], {{.*}} : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: @_Z13cmp_bool_uint
// LLVM: %[[PTR:.*]] = load ptr
// LLVM: %[[CMP:.*]] = load i32
// LLVM: %[[UPD:.*]] = load i32
// LLVM: %[[RES:.*]] = cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[UPD]] seq_cst seq_cst, align 4
// LLVM: %[[TMP:.*]] = extractvalue { i32, i1 } %[[RES]], 1
// LLVM: %[[EXT:.*]] = zext i1 %[[TMP]] to i8
// LLVM: store i8 %[[EXT]], ptr {{.*}}
void cmp_bool_uint(unsigned int* p, int x, int u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z15cmp_bool_ushort
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!u16i>, {{.*}} : !u16i, {{.*}} : !u16i, success = seq_cst, failure = seq_cst) syncscope(system) align(2) : (!u16i, !cir.bool)

// LLVM-LABEL: @_Z15cmp_bool_ushort
// LLVM: cmpxchg ptr {{.*}}, i16 {{.*}}, i16 {{.*}} seq_cst seq_cst, align 2
void cmp_bool_ushort(unsigned short* p, short x, short u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z14cmp_bool_ulong
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!u64i>, {{.*}} : !u64i, {{.*}} : !u64i, success = seq_cst, failure = seq_cst) syncscope(system) align(8) : (!u64i, !cir.bool)

// LLVM-LABEL: @_Z14cmp_bool_ulong
// LLVM: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8
void cmp_bool_ulong(unsigned long* p, long x, long u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z12cmp_val_uint
// CHECK: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CHECK: %[[CMP:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[CMP_U:.*]] = cir.cast(integral, %[[CMP]] : !s32i), !u32i
// CHECK: %[[UPD:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[UPD_U:.*]] = cir.cast(integral, %[[UPD]] : !s32i), !u32i
// CHECK: %[[OLD:.*]], %[[RES:.*]] = cir.atomic.cmp_xchg(%[[PTR]] : !cir.ptr<!u32i>, %[[CMP_U]] :
// CHECK-SAME: !u32i, %[[UPD_U]] : !u32i, success = seq_cst, failure = seq_cst) syncscope(system) align(4) : (!u32i, !cir.bool)
// CHECK: %[[R:.*]] = cir.cast(integral, %[[OLD]] : !u32i), !s32i
// CHECK: cir.store %[[R]], {{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: @_Z12cmp_val_uint
// LLVM: %[[PTR:.*]] = load ptr
// LLVM: %[[CMP:.*]] = load i32
// LLVM: %[[UPD:.*]] = load i32
// LLVM: %[[RES:.*]] = cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[UPD]] seq_cst seq_cst, align 4
// LLVM: %[[TMP:.*]] = extractvalue { i32, i1 } %[[RES]], 0
// LLVM: store i32 %[[TMP]], ptr {{.*}}
void cmp_val_uint(unsigned int* p, int x, int u) {
  int r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z14cmp_val_ushort
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!u16i>, {{.*}} : !u16i, {{.*}} : !u16i, success = seq_cst, failure = seq_cst) syncscope(system) align(2) : (!u16i, !cir.bool)

// LLVM-LABEL: @_Z14cmp_val_ushort
// LLVM: cmpxchg ptr {{.*}}, i16 {{.*}}, i16 {{.*}} seq_cst seq_cst, align 2
void cmp_val_ushort(unsigned short* p, short x, short u) {
  short r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z13cmp_val_ulong
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!u64i>, {{.*}} : !u64i, {{.*}} : !u64i, success = seq_cst, failure = seq_cst) syncscope(system) align(8) : (!u64i, !cir.bool)

// LLVM-LABEL: @_Z13cmp_val_ulong
// LLVM: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8
void cmp_val_ulong(unsigned long* p, long x, long u) {
  long r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @test_op_and_fetch
// LLVM-LABEL: @test_op_and_fetch
extern "C" void test_op_and_fetch(void)
{
  // CHECK: [[VAL0:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s8i
  // CHECK: [[RES0:%.*]] = cir.atomic.fetch(add, {{%.*}} : !cir.ptr<!s8i>, [[VAL0]] : !s8i, seq_cst) fetch_first : !s8i
  // CHECK: [[RET0:%.*]] = cir.binop(add, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw add ptr @sc, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = add i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr @sc, align 1
  sc = __sync_add_and_fetch(&sc, uc);

  // CHECK: [[RES1:%.*]] = cir.atomic.fetch(add, {{%.*}} : !cir.ptr<!u8i>, [[VAL1:%.*]] : !u8i, seq_cst) fetch_first : !u8i
  // CHECK: [[RET1:%.*]] = cir.binop(add, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw add ptr @uc, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = add i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr @uc, align 1
  uc = __sync_add_and_fetch(&uc, uc);
  
  // CHECK: [[VAL2:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s16i
  // CHECK: [[RES2:%.*]] = cir.atomic.fetch(add, {{%.*}} : !cir.ptr<!s16i>, [[VAL2]] : !s16i, seq_cst) fetch_first : !s16i
  // CHECK: [[RET2:%.*]] = cir.binop(add, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw add ptr @ss, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = add i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr @ss, align 2
  ss = __sync_add_and_fetch(&ss, uc);

  // CHECK: [[VAL3:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u16i
  // CHECK: [[RES3:%.*]] = cir.atomic.fetch(add, {{%.*}} : !cir.ptr<!u16i>, [[VAL3]] : !u16i, seq_cst) fetch_first : !u16i
  // CHECK: [[RET3:%.*]] = cir.binop(add, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw add ptr @us, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = add i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr @us
  us = __sync_add_and_fetch(&us, uc);

  // CHECK: [[VAL4:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s32i
  // CHECK: [[RES4:%.*]] = cir.atomic.fetch(add, {{%.*}} : !cir.ptr<!s32i>, [[VAL4]] : !s32i, seq_cst) fetch_first : !s32i
  // CHECK: [[RET4:%.*]] = cir.binop(add, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw add ptr @si, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = add i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr @si, align 4
  si = __sync_add_and_fetch(&si, uc);

  // CHECK: [[VAL5:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u32i
  // CHECK: [[RES5:%.*]] = cir.atomic.fetch(add, {{%.*}} : !cir.ptr<!u32i>, [[VAL5]] : !u32i, seq_cst) fetch_first : !u32i
  // CHECK: [[RET5:%.*]] = cir.binop(add, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw add ptr @ui, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = add i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr @ui, align 4
  ui = __sync_add_and_fetch(&ui, uc);

  // CHECK: [[VAL6:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s64i
  // CHECK: [[RES6:%.*]] = cir.atomic.fetch(add, {{%.*}} : !cir.ptr<!s64i>, [[VAL6]] : !s64i, seq_cst) fetch_first : !s64i
  // CHECK: [[RET6:%.*]] = cir.binop(add, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw add ptr @sll, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = add i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr @sll, align 8
  sll = __sync_add_and_fetch(&sll, uc);

  // CHECK: [[VAL7:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u64i
  // CHECK: [[RES7:%.*]] = cir.atomic.fetch(add, {{%.*}} : !cir.ptr<!u64i>, [[VAL7]] : !u64i, seq_cst) fetch_first : !u64i
  // CHECK: [[RET7:%.*]] = cir.binop(add, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw add ptr @ull, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = add i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr @ull, align 8
  ull = __sync_add_and_fetch(&ull, uc);

  // CHECK: [[VAL0:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s8i
  // CHECK: [[RES0:%.*]] = cir.atomic.fetch(sub, {{%.*}} : !cir.ptr<!s8i>, [[VAL0]] : !s8i, seq_cst) fetch_first : !s8i
  // CHECK: [[RET0:%.*]] = cir.binop(sub, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw sub ptr @sc, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = sub i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr @sc, align 1
  sc = __sync_sub_and_fetch(&sc, uc);

  // CHECK: [[RES1:%.*]] = cir.atomic.fetch(sub, {{%.*}} : !cir.ptr<!u8i>, [[VAL1:%.*]] : !u8i, seq_cst) fetch_first : !u8i
  // CHECK: [[RET1:%.*]] = cir.binop(sub, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw sub ptr @uc, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = sub i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr @uc, align 1
  uc = __sync_sub_and_fetch(&uc, uc);
  
  // CHECK: [[VAL2:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s16i
  // CHECK: [[RES2:%.*]] = cir.atomic.fetch(sub, {{%.*}} : !cir.ptr<!s16i>, [[VAL2]] : !s16i, seq_cst) fetch_first : !s16i
  // CHECK: [[RET2:%.*]] = cir.binop(sub, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw sub ptr @ss, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = sub i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr @ss, align 2
  ss = __sync_sub_and_fetch(&ss, uc);

  // CHECK: [[VAL3:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u16i
  // CHECK: [[RES3:%.*]] = cir.atomic.fetch(sub, {{%.*}} : !cir.ptr<!u16i>, [[VAL3]] : !u16i, seq_cst) fetch_first : !u16i
  // CHECK: [[RET3:%.*]] = cir.binop(sub, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw sub ptr @us, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = sub i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr @us
  us = __sync_sub_and_fetch(&us, uc);

  // CHECK: [[VAL4:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s32i
  // CHECK: [[RES4:%.*]] = cir.atomic.fetch(sub, {{%.*}} : !cir.ptr<!s32i>, [[VAL4]] : !s32i, seq_cst) fetch_first : !s32i
  // CHECK: [[RET4:%.*]] = cir.binop(sub, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw sub ptr @si, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = sub i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr @si, align 4
  si = __sync_sub_and_fetch(&si, uc);

  // CHECK: [[VAL5:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u32i
  // CHECK: [[RES5:%.*]] = cir.atomic.fetch(sub, {{%.*}} : !cir.ptr<!u32i>, [[VAL5]] : !u32i, seq_cst) fetch_first : !u32i
  // CHECK: [[RET5:%.*]] = cir.binop(sub, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw sub ptr @ui, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = sub i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr @ui, align 4
  ui = __sync_sub_and_fetch(&ui, uc);

  // CHECK: [[VAL6:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s64i
  // CHECK: [[RES6:%.*]] = cir.atomic.fetch(sub, {{%.*}} : !cir.ptr<!s64i>, [[VAL6]] : !s64i, seq_cst) fetch_first : !s64i
  // CHECK: [[RET6:%.*]] = cir.binop(sub, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw sub ptr @sll, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = sub i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr @sll, align 8
  sll = __sync_sub_and_fetch(&sll, uc);

  // CHECK: [[VAL7:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u64i
  // CHECK: [[RES7:%.*]] = cir.atomic.fetch(sub, {{%.*}} : !cir.ptr<!u64i>, [[VAL7]] : !u64i, seq_cst) fetch_first : !u64i
  // CHECK: [[RET7:%.*]] = cir.binop(sub, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw sub ptr @ull, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = sub i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr @ull, align 8
  ull = __sync_sub_and_fetch(&ull, uc);

  // CHECK: [[VAL0:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s8i
  // CHECK: [[RES0:%.*]] = cir.atomic.fetch(and, {{%.*}} : !cir.ptr<!s8i>, [[VAL0]] : !s8i, seq_cst) fetch_first : !s8i
  // CHECK: [[RET0:%.*]] = cir.binop(and, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw and ptr @sc, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = and i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr @sc, align 1
  sc = __sync_and_and_fetch(&sc, uc);

  // CHECK: [[RES1:%.*]] = cir.atomic.fetch(and, {{%.*}} : !cir.ptr<!u8i>, [[VAL1:%.*]] : !u8i, seq_cst) fetch_first : !u8i
  // CHECK: [[RET1:%.*]] = cir.binop(and, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw and ptr @uc, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = and i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr @uc, align 1
  uc = __sync_and_and_fetch(&uc, uc);
  
  // CHECK: [[VAL2:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s16i
  // CHECK: [[RES2:%.*]] = cir.atomic.fetch(and, {{%.*}} : !cir.ptr<!s16i>, [[VAL2]] : !s16i, seq_cst) fetch_first : !s16i
  // CHECK: [[RET2:%.*]] = cir.binop(and, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw and ptr @ss, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = and i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr @ss, align 2
  ss = __sync_and_and_fetch(&ss, uc);

  // CHECK: [[VAL3:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u16i
  // CHECK: [[RES3:%.*]] = cir.atomic.fetch(and, {{%.*}} : !cir.ptr<!u16i>, [[VAL3]] : !u16i, seq_cst) fetch_first : !u16i
  // CHECK: [[RET3:%.*]] = cir.binop(and, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw and ptr @us, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = and i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr @us
  us = __sync_and_and_fetch(&us, uc);

  // CHECK: [[VAL4:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s32i
  // CHECK: [[RES4:%.*]] = cir.atomic.fetch(and, {{%.*}} : !cir.ptr<!s32i>, [[VAL4]] : !s32i, seq_cst) fetch_first : !s32i
  // CHECK: [[RET4:%.*]] = cir.binop(and, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw and ptr @si, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = and i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr @si, align 4
  si = __sync_and_and_fetch(&si, uc);

  // CHECK: [[VAL5:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u32i
  // CHECK: [[RES5:%.*]] = cir.atomic.fetch(and, {{%.*}} : !cir.ptr<!u32i>, [[VAL5]] : !u32i, seq_cst) fetch_first : !u32i
  // CHECK: [[RET5:%.*]] = cir.binop(and, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw and ptr @ui, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = and i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr @ui, align 4
  ui = __sync_and_and_fetch(&ui, uc);

  // CHECK: [[VAL6:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s64i
  // CHECK: [[RES6:%.*]] = cir.atomic.fetch(and, {{%.*}} : !cir.ptr<!s64i>, [[VAL6]] : !s64i, seq_cst) fetch_first : !s64i
  // CHECK: [[RET6:%.*]] = cir.binop(and, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw and ptr @sll, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = and i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr @sll, align 8
  sll = __sync_and_and_fetch(&sll, uc);

  // CHECK: [[VAL7:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u64i
  // CHECK: [[RES7:%.*]] = cir.atomic.fetch(and, {{%.*}} : !cir.ptr<!u64i>, [[VAL7]] : !u64i, seq_cst) fetch_first : !u64i
  // CHECK: [[RET7:%.*]] = cir.binop(and, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw and ptr @ull, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = and i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr @ull, align 8
  ull = __sync_and_and_fetch(&ull, uc);

  // CHECK: [[VAL0:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s8i
  // CHECK: [[RES0:%.*]] = cir.atomic.fetch(or, {{%.*}} : !cir.ptr<!s8i>, [[VAL0]] : !s8i, seq_cst) fetch_first : !s8i
  // CHECK: [[RET0:%.*]] = cir.binop(or, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw or ptr @sc, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = or i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr @sc, align 1
  sc = __sync_or_and_fetch(&sc, uc);

  // CHECK: [[RES1:%.*]] = cir.atomic.fetch(or, {{%.*}} : !cir.ptr<!u8i>, [[VAL1:%.*]] : !u8i, seq_cst) fetch_first : !u8i
  // CHECK: [[RET1:%.*]] = cir.binop(or, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw or ptr @uc, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = or i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr @uc, align 1
  uc = __sync_or_and_fetch(&uc, uc);
  
  // CHECK: [[VAL2:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s16i
  // CHECK: [[RES2:%.*]] = cir.atomic.fetch(or, {{%.*}} : !cir.ptr<!s16i>, [[VAL2]] : !s16i, seq_cst) fetch_first : !s16i
  // CHECK: [[RET2:%.*]] = cir.binop(or, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw or ptr @ss, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = or i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr @ss, align 2
  ss = __sync_or_and_fetch(&ss, uc);

  // CHECK: [[VAL3:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u16i
  // CHECK: [[RES3:%.*]] = cir.atomic.fetch(or, {{%.*}} : !cir.ptr<!u16i>, [[VAL3]] : !u16i, seq_cst) fetch_first : !u16i
  // CHECK: [[RET3:%.*]] = cir.binop(or, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw or ptr @us, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = or i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr @us
  us = __sync_or_and_fetch(&us, uc);

  // CHECK: [[VAL4:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s32i
  // CHECK: [[RES4:%.*]] = cir.atomic.fetch(or, {{%.*}} : !cir.ptr<!s32i>, [[VAL4]] : !s32i, seq_cst) fetch_first : !s32i
  // CHECK: [[RET4:%.*]] = cir.binop(or, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw or ptr @si, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = or i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr @si, align 4
  si = __sync_or_and_fetch(&si, uc);

  // CHECK: [[VAL5:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u32i
  // CHECK: [[RES5:%.*]] = cir.atomic.fetch(or, {{%.*}} : !cir.ptr<!u32i>, [[VAL5]] : !u32i, seq_cst) fetch_first : !u32i
  // CHECK: [[RET5:%.*]] = cir.binop(or, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw or ptr @ui, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = or i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr @ui, align 4
  ui = __sync_or_and_fetch(&ui, uc);

  // CHECK: [[VAL6:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s64i
  // CHECK: [[RES6:%.*]] = cir.atomic.fetch(or, {{%.*}} : !cir.ptr<!s64i>, [[VAL6]] : !s64i, seq_cst) fetch_first : !s64i
  // CHECK: [[RET6:%.*]] = cir.binop(or, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw or ptr @sll, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = or i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr @sll, align 8
  sll = __sync_or_and_fetch(&sll, uc);

  // CHECK: [[VAL7:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u64i
  // CHECK: [[RES7:%.*]] = cir.atomic.fetch(or, {{%.*}} : !cir.ptr<!u64i>, [[VAL7]] : !u64i, seq_cst) fetch_first : !u64i
  // CHECK: [[RET7:%.*]] = cir.binop(or, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw or ptr @ull, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = or i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr @ull, align 8
  ull = __sync_or_and_fetch(&ull, uc);

  // CHECK: [[VAL0:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s8i
  // CHECK: [[RES0:%.*]] = cir.atomic.fetch(xor, {{%.*}} : !cir.ptr<!s8i>, [[VAL0]] : !s8i, seq_cst) fetch_first : !s8i
  // CHECK: [[RET0:%.*]] = cir.binop(xor, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw xor ptr @sc, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = xor i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr @sc, align 1
  sc = __sync_xor_and_fetch(&sc, uc);

  // CHECK: [[RES1:%.*]] = cir.atomic.fetch(xor, {{%.*}} : !cir.ptr<!u8i>, [[VAL1:%.*]] : !u8i, seq_cst) fetch_first : !u8i
  // CHECK: [[RET1:%.*]] = cir.binop(xor, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw xor ptr @uc, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = xor i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr @uc, align 1
  uc = __sync_xor_and_fetch(&uc, uc);
  
  // CHECK: [[VAL2:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s16i
  // CHECK: [[RES2:%.*]] = cir.atomic.fetch(xor, {{%.*}} : !cir.ptr<!s16i>, [[VAL2]] : !s16i, seq_cst) fetch_first : !s16i
  // CHECK: [[RET2:%.*]] = cir.binop(xor, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw xor ptr @ss, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = xor i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr @ss, align 2
  ss = __sync_xor_and_fetch(&ss, uc);

  // CHECK: [[VAL3:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u16i
  // CHECK: [[RES3:%.*]] = cir.atomic.fetch(xor, {{%.*}} : !cir.ptr<!u16i>, [[VAL3]] : !u16i, seq_cst) fetch_first : !u16i
  // CHECK: [[RET3:%.*]] = cir.binop(xor, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw xor ptr @us, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = xor i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr @us
  us = __sync_xor_and_fetch(&us, uc);

  // CHECK: [[VAL4:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s32i
  // CHECK: [[RES4:%.*]] = cir.atomic.fetch(xor, {{%.*}} : !cir.ptr<!s32i>, [[VAL4]] : !s32i, seq_cst) fetch_first : !s32i
  // CHECK: [[RET4:%.*]] = cir.binop(xor, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw xor ptr @si, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = xor i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr @si, align 4
  si = __sync_xor_and_fetch(&si, uc);

  // CHECK: [[VAL5:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u32i
  // CHECK: [[RES5:%.*]] = cir.atomic.fetch(xor, {{%.*}} : !cir.ptr<!u32i>, [[VAL5]] : !u32i, seq_cst) fetch_first : !u32i
  // CHECK: [[RET5:%.*]] = cir.binop(xor, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw xor ptr @ui, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = xor i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr @ui, align 4
  ui = __sync_xor_and_fetch(&ui, uc);

  // CHECK: [[VAL6:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s64i
  // CHECK: [[RES6:%.*]] = cir.atomic.fetch(xor, {{%.*}} : !cir.ptr<!s64i>, [[VAL6]] : !s64i, seq_cst) fetch_first : !s64i
  // CHECK: [[RET6:%.*]] = cir.binop(xor, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw xor ptr @sll, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = xor i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr @sll, align 8
  sll = __sync_xor_and_fetch(&sll, uc);

  // CHECK: [[VAL7:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u64i
  // CHECK: [[RES7:%.*]] = cir.atomic.fetch(xor, {{%.*}} : !cir.ptr<!u64i>, [[VAL7]] : !u64i, seq_cst) fetch_first : !u64i
  // CHECK: [[RET7:%.*]] = cir.binop(xor, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw xor ptr @ull, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = xor i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr @ull, align 8
  ull = __sync_xor_and_fetch(&ull, uc);

  // CHECK: [[VAL0:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s8i
  // CHECK: [[RES0:%.*]] = cir.atomic.fetch(nand, {{%.*}} : !cir.ptr<!s8i>, [[VAL0]] : !s8i, seq_cst) fetch_first : !s8i
  // CHECK: [[INTERM0:%.*]] = cir.binop(and, [[RES0]], [[VAL0]]) : !s8i
  // CHECK: [[RET0:%.*]] =  cir.unary(not, [[INTERM0]]) : !s8i, !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw nand ptr @sc, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[INTERM0:%.*]] = and i8 [[RES0]], [[VAL0]]
  // LLVM:  [[RET0:%.*]] = xor i8 [[INTERM0]], -1
  // LLVM:  store i8 [[RET0]], ptr @sc, align 1
  sc = __sync_nand_and_fetch(&sc, uc);

  // CHECK: [[RES1:%.*]] = cir.atomic.fetch(nand, {{%.*}} : !cir.ptr<!u8i>, [[VAL1:%.*]] : !u8i, seq_cst) fetch_first : !u8i
  // CHECK: [[INTERM1:%.*]] = cir.binop(and, [[RES1]], [[VAL1]]) : !u8i
  // CHECK: [[RET1:%.*]] = cir.unary(not, [[INTERM1]]) : !u8i, !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw nand ptr @uc, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[INTERM1:%.*]] = and i8 [[RES1]], [[VAL1]]
  // LLVM:  [[RET1:%.*]] = xor i8 [[INTERM1]], -1
  // LLVM:  store i8 [[RET1]], ptr @uc, align 1
  uc = __sync_nand_and_fetch(&uc, uc);
  
  // CHECK: [[VAL2:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s16i
  // CHECK: [[RES2:%.*]] = cir.atomic.fetch(nand, {{%.*}} : !cir.ptr<!s16i>, [[VAL2]] : !s16i, seq_cst) fetch_first : !s16i
  // CHECK: [[INTERM2:%.*]] = cir.binop(and, [[RES2]], [[VAL2]]) : !s16i
  // CHECK: [[RET2:%.*]] =  cir.unary(not, [[INTERM2]]) : !s16i, !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw nand ptr @ss, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[INTERM2:%.*]] = and i16 [[RES2]], [[CONV2]]
  // LLVM:  [[RET2:%.*]] = xor i16 [[INTERM2]], -1
  // LLVM:  store i16 [[RET2]], ptr @ss, align 2
  ss = __sync_nand_and_fetch(&ss, uc);

  // CHECK: [[VAL3:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u16i
  // CHECK: [[RES3:%.*]] = cir.atomic.fetch(nand, {{%.*}} : !cir.ptr<!u16i>, [[VAL3]] : !u16i, seq_cst) fetch_first : !u16i
  // CHECK: [[INTERM3:%.*]] = cir.binop(and, [[RES3]], [[VAL3]]) : !u16i
  // CHECK: [[RET3:%.*]] =  cir.unary(not, [[INTERM3]]) : !u16i, !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw nand ptr @us, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[INTERM3:%.*]] = and i16 [[RES3]], [[CONV3]]
  // LLVM:  [[RET3:%.*]] = xor i16 [[INTERM3]], -1
  // LLVM:  store i16 [[RET3]], ptr @us, align 2
  us = __sync_nand_and_fetch(&us, uc);

  // CHECK: [[VAL4:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s32i
  // CHECK: [[RES4:%.*]] = cir.atomic.fetch(nand, {{%.*}} : !cir.ptr<!s32i>, [[VAL4]] : !s32i, seq_cst) fetch_first : !s32i
  // CHECK: [[INTERM4:%.*]] = cir.binop(and, [[RES4]], [[VAL4]]) : !s32i
  // CHECK: [[RET4:%.*]] =  cir.unary(not, [[INTERM4]]) : !s32i, !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw nand ptr @si, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[INTERM4:%.*]] = and i32 [[RES4]], [[CONV4]]
  // LLVM:  [[RET4:%.*]] = xor i32 [[INTERM4]], -1
  // LLVM:  store i32 [[RET4]], ptr @si, align 4
  si = __sync_nand_and_fetch(&si, uc);

  // CHECK: [[VAL5:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u32i
  // CHECK: [[RES5:%.*]] = cir.atomic.fetch(nand, {{%.*}} : !cir.ptr<!u32i>, [[VAL5]] : !u32i, seq_cst) fetch_first : !u32i
  // CHECK: [[INTERM5:%.*]] = cir.binop(and, [[RES5]], [[VAL5]]) : !u32i
  // CHECK: [[RET5:%.*]] =  cir.unary(not, [[INTERM5]]) : !u32i, !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw nand ptr @ui, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[INTERM5:%.*]] = and i32 [[RES5]], [[CONV5]]
  // LLVM:  [[RET5:%.*]] = xor i32 [[INTERM5]], -1
  // LLVM:  store i32 [[RET5]], ptr @ui, align 4
  ui = __sync_nand_and_fetch(&ui, uc);

  // CHECK: [[VAL6:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !s64i
  // CHECK: [[RES6:%.*]] = cir.atomic.fetch(nand, {{%.*}} : !cir.ptr<!s64i>, [[VAL6]] : !s64i, seq_cst) fetch_first : !s64i
  // CHECK: [[INTERM6:%.*]] = cir.binop(and, [[RES6]], [[VAL6]]) : !s64i
  // CHECK: [[RET6:%.*]] =  cir.unary(not, [[INTERM6]]) : !s64i, !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw nand ptr @sll, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[INTERM6:%.*]] = and i64 [[RES6]], [[CONV6]]
  // LLVM:  [[RET6:%.*]] = xor i64 [[INTERM6]], -1
  // LLVM:  store i64 [[RET6]], ptr @sll, align 8
  sll = __sync_nand_and_fetch(&sll, uc);

  // CHECK: [[VAL7:%.*]] = cir.cast(integral, {{%.*}} : !u8i), !u64i
  // CHECK: [[RES7:%.*]] = cir.atomic.fetch(nand, {{%.*}} : !cir.ptr<!u64i>, [[VAL7]] : !u64i, seq_cst) fetch_first : !u64i
  // CHECK: [[INTERM7:%.*]] = cir.binop(and, [[RES7]], [[VAL7]]) : !u64i
  // CHECK: [[RET7:%.*]] =  cir.unary(not, [[INTERM7]]) : !u64i, !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr @uc, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw nand ptr @ull, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[INTERM7:%.*]] = and i64 [[RES7]], [[CONV7]]
  // LLVM:  [[RET7:%.*]] = xor i64 [[INTERM7]], -1
  // LLVM:  store i64 [[RET7]], ptr @ull, align 8
  ull = __sync_nand_and_fetch(&ull, uc);
}
