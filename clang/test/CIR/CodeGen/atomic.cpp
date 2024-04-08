// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef struct _a {
  _Atomic(int) d;
} at;

void m() { at y; }

// CHECK: ![[A:.*]] = !cir.struct<struct "_a" {!cir.int<s, 32>}>

enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
};

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
// CHECK:  cir.atomic.binop_fetch(add, %[[I]] : !cir.ptr<!s32i>, %[[VAL]] : !s32i, seq_cst) : !s32i

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
// CHECK: cir.atomic.binop_fetch(sub, {{.*}}, relaxed
// CHECK: cir.atomic.binop_fetch(and, {{.*}}, acquire
// CHECK: cir.atomic.binop_fetch(or, {{.*}}, acquire
// CHECK: cir.atomic.binop_fetch(xor, {{.*}}, release

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
// CHECK: cir.atomic.binop_fetch(nand, {{.*}}, acq_rel

// LLVM: define i32 @_Z16nand_binop_fetchPi
// LLVM: %[[RMW_NAND:.*]] = atomicrmw nand ptr {{.*}} acq_rel
// LLVM: %[[AND:.*]] = and i32 %[[RMW_NAND]]
// LLVM: = xor i32 %[[AND]], -1

int fp_binop_fetch(float *i) {
  __atomic_add_fetch(i, 1, memory_order_seq_cst);
  return __atomic_sub_fetch(i, 1, memory_order_seq_cst);
}

// CHECK: cir.func @_Z14fp_binop_fetchPf
// CHECK: cir.atomic.binop_fetch(add,
// CHECK: cir.atomic.binop_fetch(sub,

// LLVM: %[[RMW_FADD:.*]] = atomicrmw fadd ptr
// LLVM: fadd float %[[RMW_FADD]]
// LLVM: %[[RMW_FSUB:.*]] = atomicrmw fsub ptr
// LLVM: fsub float %[[RMW_FSUB]]