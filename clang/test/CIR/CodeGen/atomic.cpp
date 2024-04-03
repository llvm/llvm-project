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

int fi3b(int *i) {
  return __atomic_add_fetch(i, 1, memory_order_seq_cst);
}

// CHECK: cir.func @_Z4fi3bPi
// CHECK:  %[[ARGI:.*]] = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["i", init] {alignment = 8 : i64}
// CHECK:  %[[ONE_ADDR:.*]] = cir.alloca !s32i, cir.ptr <!s32i>, [".atomictmp"] {alignment = 4 : i64}
// CHECK:  cir.store %arg0, %[[ARGI]] : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK:  %[[I:.*]] = cir.load %[[ARGI]] : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:  %[[ONE:.*]] = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:  cir.store %[[ONE]], %[[ONE_ADDR]] : !s32i, cir.ptr <!s32i>
// CHECK:  %[[VAL:.*]] = cir.load %[[ONE_ADDR]] : cir.ptr <!s32i>, !s32i
// CHECK:  cir.atomic.add_fetch(%[[I]] : !cir.ptr<!s32i>, %[[VAL]] : !s32i, seq_cst) : !s32i

// LLVM: define i32 @_Z4fi3bPi
// LLVM: %[[RMW:.*]] = atomicrmw add ptr {{.*}}, i32 %[[VAL:.*]] seq_cst, align 4
// LLVM: add i32 %[[RMW]], %[[VAL]]