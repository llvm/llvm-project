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

// CHECK: ![[A:.*]] = !cir.struct<struct "_a" {!cir.int<s, 32>}>

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
// CHECK: cir.atomic.xchg({{.*}} : !cir.ptr<!ty_S>, {{.*}} : !u64i, seq_cst) : !u64i

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
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!ty_S>, {{.*}} : !u64i, {{.*}} : !u64i, success = seq_cst, failure = seq_cst) weak : (!u64i, !cir.bool)

// LLVM-LABEL: @_Z3fd4P1SS0_S0_
// LLVM: cmpxchg weak ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8

bool fi4a(int *i) {
  int cmp = 0;
  int desired = 1;
  return __atomic_compare_exchange(i, &cmp, &desired, false, memory_order_acquire, memory_order_acquire);
}

// CHECK-LABEL: @_Z4fi4aPi
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s32i>, {{.*}} : !s32i, {{.*}} : !s32i, success = acquire, failure = acquire) : (!s32i, !cir.bool)

// LLVM-LABEL: @_Z4fi4aPi
// LLVM: %[[RES:.*]] = cmpxchg ptr %7, i32 %8, i32 %9 acquire acquire, align 4
// LLVM: extractvalue { i32, i1 } %[[RES]], 0
// LLVM: extractvalue { i32, i1 } %[[RES]], 1

bool fi4b(int *i) {
  int cmp = 0;
  return __atomic_compare_exchange_n(i, &cmp, 1, true, memory_order_acquire, memory_order_acquire);
}

// CHECK-LABEL: @_Z4fi4bPi
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s32i>, {{.*}} : !s32i, {{.*}} : !s32i, success = acquire, failure = acquire) weak : (!s32i, !cir.bool)

// LLVM-LABEL: @_Z4fi4bPi
// LLVM: %[[R:.*]] = cmpxchg weak ptr {{.*}}, i32 {{.*}}, i32 {{.*}} acquire acquire, align 4
// LLVM: extractvalue { i32, i1 } %[[R]], 0
// LLVM: extractvalue { i32, i1 } %[[R]], 1

bool fi4c(atomic_int *i) {
  int cmp = 0;
  return atomic_compare_exchange_strong(i, &cmp, 1);
}

// CHECK-LABEL: @_Z4fi4cPU7_Atomici
// CHECK: %old, %cmp = cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s32i>, {{.*}} : !s32i, {{.*}} : !s32i, success = seq_cst, failure = seq_cst) : (!s32i, !cir.bool)
// CHECK: %[[CMP:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK: cir.if %[[CMP:.*]] {
// CHECK:   cir.store %old, {{.*}} : !s32i, !cir.ptr<!s32i>
// CHECK: }

// LLVM-LABEL: @_Z4fi4cPU7_Atomici
// LLVM: cmpxchg ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4

bool fsb(bool *c) {
  return __atomic_exchange_n(c, 1, memory_order_seq_cst);
}

// CHECK-LABEL: @_Z3fsbPb
// CHECK: cir.atomic.xchg({{.*}} : !cir.ptr<!cir.bool>, {{.*}} : !u8i, seq_cst) : !u8i

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


// CHECK-LABEL: @_Z8inc_long
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!s64i>, {{.*}} : !s64i, seq_cst) fetch_first : !s64i

// LLVM-LABEL: @_Z8inc_long
// LLVM: atomicrmw add ptr {{.*}}, i64 {{.*}} seq_cst, align 8

void inc_long(long* a, long b) {
  long c = __sync_fetch_and_add(a, 2);
}

// CHECK-LABEL: @_Z9inc_short
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!s16i>, {{.*}} : !s16i, seq_cst) fetch_first : !s16i

// LLVM-LABEL: @_Z9inc_short
// LLVM: atomicrmw add ptr {{.*}}, i16 {{.*}} seq_cst, align 2
void inc_short(short* a, short b) {
  short c = __sync_fetch_and_add(a, 2);
}

// CHECK-LABEL: @_Z8inc_byte
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!s8i>, {{.*}} : !s8i, seq_cst) fetch_first : !s8i

// LLVM-LABEL: @_Z8inc_byte
// LLVM: atomicrmw add ptr {{.*}}, i8 {{.*}} seq_cst, align 1
void inc_byte(char* a, char b) {
  char c = __sync_fetch_and_add(a, b);
}


// CHECK-LABEL: @_Z12cmp_bool_int
// CHECK: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[CMP:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[UPD:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[OLD:.*]], %[[RES:.*]] = cir.atomic.cmp_xchg(%[[PTR]] : !cir.ptr<!s32i>, %[[CMP]] : !s32i, %[[UPD]] : !s32i, success = seq_cst, failure = seq_cst) : (!s32i, !cir.bool)
// CHECK: cir.store %[[RES]], {{.*}} : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: @_Z12cmp_bool_int
// LLVM: %[[PTR:.*]] = load ptr
// LLVM: %[[CMP:.*]] = load i32
// LLVM: %[[UPD:.*]] = load i32
// LLVM: %[[RES:.*]] = cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[UPD]] seq_cst seq_cst
// LLVM: %[[TMP:.*]] = extractvalue { i32, i1 } %[[RES]], 1
// LLVM: %[[EXT:.*]] = zext i1 %[[TMP]] to i8
// LLVM: store i8 %[[EXT]], ptr {{.*}}
void cmp_bool_int(int* p, int x, int u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z13cmp_bool_long
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s64i>, {{.*}} : !s64i, {{.*}} : !s64i, success = seq_cst, failure = seq_cst) : (!s64i, !cir.bool)

// LLVM-LABEL: @_Z13cmp_bool_long
// LLVM: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst
void cmp_bool_long(long* p, long x, long u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z14cmp_bool_short
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s16i>, {{.*}} : !s16i, {{.*}} : !s16i, success = seq_cst, failure = seq_cst) : (!s16i, !cir.bool)

// LLVM-LABEL: @_Z14cmp_bool_short
// LLVM: cmpxchg ptr {{.*}}, i16 {{.*}}, i16 {{.*}} seq_cst seq_cst
void cmp_bool_short(short* p, short x, short u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z13cmp_bool_byte
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s8i>, {{.*}} : !s8i, {{.*}} : !s8i, success = seq_cst, failure = seq_cst) : (!s8i, !cir.bool)

// LLVM-LABEL: @_Z13cmp_bool_byte
// LLVM: cmpxchg ptr {{.*}}, i8 {{.*}}, i8 {{.*}} seq_cst seq_cst
void cmp_bool_byte(char* p, char x, char u) {
  bool r = __sync_bool_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z11cmp_val_int
// CHECK: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[CMP:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[UPD:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[OLD:.*]], %[[RES:.*]] = cir.atomic.cmp_xchg(%[[PTR]] : !cir.ptr<!s32i>, %[[CMP]] : !s32i, %[[UPD]] : !s32i, success = seq_cst, failure = seq_cst) : (!s32i, !cir.bool)
// CHECK: cir.store %[[OLD]], {{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: @_Z11cmp_val_int
// LLVM: %[[PTR:.*]] = load ptr
// LLVM: %[[CMP:.*]] = load i32
// LLVM: %[[UPD:.*]] = load i32
// LLVM: %[[RES:.*]] = cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[UPD]] seq_cst seq_cst
// LLVM: %[[TMP:.*]] = extractvalue { i32, i1 } %[[RES]], 0
// LLVM: store i32 %[[TMP]], ptr {{.*}}
void cmp_val_int(int* p, int x, int u) {
  int r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z12cmp_val_long
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s64i>, {{.*}} : !s64i, {{.*}} : !s64i, success = seq_cst, failure = seq_cst) : (!s64i, !cir.bool)

// LLVM-LABEL: @_Z12cmp_val_long
// LLVM: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst
void cmp_val_long(long* p, long x, long u) {
  long r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z13cmp_val_short
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s16i>, {{.*}} : !s16i, {{.*}} : !s16i, success = seq_cst, failure = seq_cst) : (!s16i, !cir.bool)

// LLVM-LABEL: @_Z13cmp_val_short
// LLVM: cmpxchg ptr {{.*}}, i16 {{.*}}, i16 {{.*}} seq_cst seq_cst
void cmp_val_short(short* p, short x, short u) {
  short r = __sync_val_compare_and_swap(p, x, u);
}

// CHECK-LABEL: @_Z12cmp_val_byte
// CHECK: cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!s8i>, {{.*}} : !s8i, {{.*}} : !s8i, success = seq_cst, failure = seq_cst) : (!s8i, !cir.bool)

// LLVM-LABEL: @_Z12cmp_val_byte
// LLVM: cmpxchg ptr {{.*}}, i8 {{.*}}, i8 {{.*}} seq_cst seq_cst
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

// CHECK-LABEL: @_Z9inc_ulong
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!u64i>, {{.*}} : !u64i, seq_cst) fetch_first : !u64i

// LLVM-LABEL: @_Z9inc_ulong
// LLVM: atomicrmw add ptr {{.*}}, i64 {{.*}} seq_cst, align 8
void inc_ulong(unsigned long* a, long b) {
  unsigned long c = __sync_fetch_and_add(a, b);
}

// CHECK-LABEL: @_Z9inc_uchar
// CHECK: cir.atomic.fetch(add, {{.*}} : !cir.ptr<!u8i>, {{.*}} : !u8i, seq_cst) fetch_first : !u8i

// LLVM-LABEL: @_Z9inc_uchar
// LLVM: atomicrmw add ptr {{.*}}, i8 {{.*}} seq_cst, align 1
void inc_uchar(unsigned char* a, char b) {
  unsigned char c = __sync_fetch_and_add(a, b);
}