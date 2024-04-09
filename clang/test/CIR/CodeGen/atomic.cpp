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
// CHECK:  %[[ARGI:.*]] = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["i", init] {alignment = 8 : i64}
// CHECK:  %[[ONE_ADDR:.*]] = cir.alloca !s32i, cir.ptr <!s32i>, [".atomictmp"] {alignment = 4 : i64}
// CHECK:  cir.store %arg0, %[[ARGI]] : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK:  %[[I:.*]] = cir.load %[[ARGI]] : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:  %[[ONE:.*]] = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:  cir.store %[[ONE]], %[[ONE_ADDR]] : !s32i, cir.ptr <!s32i>
// CHECK:  %[[VAL:.*]] = cir.load %[[ONE_ADDR]] : cir.ptr <!s32i>, !s32i
// CHECK:  cir.atomic.fetch(add, %[[I]] : !cir.ptr<!s32i>, %[[VAL]] : !s32i, seq_cst) : !s32i

// LLVM: define i32 @_Z17basic_binop_fetchPi
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

// LLVM: define i32 @_Z17other_binop_fetchPi
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

// LLVM: define i32 @_Z16nand_binop_fetchPi
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

// LLVM: define i32 @_Z14fp_binop_fetchPf
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

// LLVM: define i32 @_Z11fetch_binopPi
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

// LLVM: define void @_Z13min_max_fetchPi
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