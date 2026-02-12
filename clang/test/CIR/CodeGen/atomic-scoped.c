// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2>%t-before-target-lowering.cir
// RUN: FileCheck --input-file=%t-before-target-lowering.cir %s --check-prefixes=CIR-BEFORE-TL
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void scoped_atomic_load(int *ptr) {
  // CIR-BEFORE-TL-LABEL: @scoped_atomic_load
  // CIR-LABEL: @scoped_atomic_load
  // LLVM-LABEL: @scoped_atomic_load
  // OGCG-LABEL: @scoped_atomic_load

  int x;
  __scoped_atomic_load(ptr, &x, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: %{{.+}} = cir.load align(4) syncscope(single_thread) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR: %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // LLVM: %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4

  __scoped_atomic_load(ptr, &x, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR: %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // LLVM: %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
}

void scoped_atomic_load_n(int *ptr) {
  // CIR-LABEL: @scoped_atomic_load_n
  // LLVM-LABEL: @scoped_atomic_load_n
  // OGCG-LABEL: @scoped_atomic_load_n

  int x;
  x = __scoped_atomic_load_n(ptr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: %{{.+}} = cir.load align(4) syncscope(single_thread) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR: %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // LLVM: %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4

  x = __scoped_atomic_load_n(ptr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR: %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // LLVM: %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
}

void scoped_atomic_store(int *ptr, int value) {
  // CIR-LABEL: @scoped_atomic_store
  // LLVM-LABEL: @scoped_atomic_store
  // OGCG-LABEL: @scoped_atomic_store

  __scoped_atomic_store(ptr, &value, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: cir.store align(4) syncscope(single_thread) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
  // LLVM: store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
  // OGCG: store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4

  __scoped_atomic_store(ptr, &value, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
  // LLVM: store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
  // OGCG: store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
}

void scoped_atomic_store_n(int *ptr, int value) {
  // CIR-LABEL: @scoped_atomic_store_n
  // LLVM-LABEL: @scoped_atomic_store_n
  // OGCG-LABEL: @scoped_atomic_store_n

  __scoped_atomic_store_n(ptr, value, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: cir.store align(4) syncscope(single_thread) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
  // LLVM: store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
  // OGCG: store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4

  __scoped_atomic_store_n(ptr, value, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
  // LLVM: store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
  // OGCG: store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
}

void scoped_atomic_exchange(int *ptr, int *value, int *old) {
  // CIR-BEFORE-TL-LABEL: @scoped_atomic_exchange
  // CIR-LABEL: @scoped_atomic_exchange
  // LLVM-LABEL: @scoped_atomic_exchange
  // OGCG-LABEL: @scoped_atomic_exchange

  __scoped_atomic_exchange(ptr, value, old, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: cir.atomic.xchg relaxed syncscope(single_thread) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg relaxed syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4

  __scoped_atomic_exchange(ptr, value, old, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: cir.atomic.xchg relaxed syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg relaxed syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
}

void scoped_atomic_exchange_n(int *ptr, int value) {
  // CIR-BEFORE-TL-LABEL: @scoped_atomic_exchange_n
  // CIR-LABEL: @scoped_atomic_exchange_n
  // LLVM-LABEL: @scoped_atomic_exchange_n
  // OGCG-LABEL: @scoped_atomic_exchange_n

  __scoped_atomic_exchange_n(ptr, value, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: cir.atomic.xchg relaxed syncscope(single_thread) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg relaxed syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4

  __scoped_atomic_exchange_n(ptr, value, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: cir.atomic.xchg relaxed syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg relaxed syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
}

void scoped_atomic_cmpxchg(int *ptr, int *expected, int *desired) {
  // CIR-BEFORE-TL-LABEL: @scoped_atomic_cmpxchg
  // CIR-LABEL: @scoped_atomic_cmpxchg
  // LLVM-LABEL: @scoped_atomic_cmpxchg
  // OGCG-LABEL: @scoped_atomic_cmpxchg

  __scoped_atomic_compare_exchange(ptr, expected, desired, /*weak=*/0,
                                   __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                   __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg success(seq_cst) failure(acquire) syncscope(single_thread) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // LLVM: %{{.+}} = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG: %{{.+}} = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4

  __scoped_atomic_compare_exchange(ptr, expected, desired, /*weak=*/1,
                                   __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                   __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) syncscope(single_thread) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // LLVM: %{{.+}} = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG: %{{.+}} = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4

  __scoped_atomic_compare_exchange(ptr, expected, desired, /*weak=*/0,
                                   __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                   __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // LLVM: %{{.+}} = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG: %{{.+}} = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4

  __scoped_atomic_compare_exchange(ptr, expected, desired, /*weak=*/1,
                                   __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                   __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // LLVM: %{{.+}} = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG: %{{.+}} = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
}

void scoped_atomic_cmpxchg_n(int *ptr, int *expected, int desired) {
  // CIR-BEFORE-TL-LABEL: @scoped_atomic_cmpxchg_n
  // CIR-LABEL: @scoped_atomic_cmpxchg_n
  // LLVM-LABEL: @scoped_atomic_cmpxchg_n
  // OGCG-LABEL: @scoped_atomic_cmpxchg_n

  __scoped_atomic_compare_exchange_n(ptr, expected, desired, /*weak=*/0,
                                     __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                     __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg success(seq_cst) failure(acquire) syncscope(single_thread) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // LLVM: %{{.+}} = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG: %{{.+}} = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4

  __scoped_atomic_compare_exchange_n(ptr, expected, desired, /*weak=*/1,
                                     __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                     __MEMORY_SCOPE_SINGLE);
  // CIR-BEFORE-TL: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) syncscope(single_thread) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // LLVM: %{{.+}} = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG: %{{.+}} = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4

  __scoped_atomic_compare_exchange_n(ptr, expected, desired, /*weak=*/0,
                                     __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                     __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // LLVM: %{{.+}} = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG: %{{.+}} = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4

  __scoped_atomic_compare_exchange_n(ptr, expected, desired, /*weak=*/1,
                                     __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                     __MEMORY_SCOPE_SYSTEM);
  // CIR-BEFORE-TL: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR: %{{.+}}, %{{.+}} = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // LLVM: %{{.+}} = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG: %{{.+}} = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
}
