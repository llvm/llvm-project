// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void f1(void) {
  _Atomic(int) x = 42;
}

// CIR-LABEL: @f1
// CIR:         %[[SLOT:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR-NEXT:    %[[INIT:.+]] = cir.const #cir.int<42> : !s32i
// CIR-NEXT:    cir.store align(4) %[[INIT]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR:       }

// LLVM-LABEL: @f1
// LLVM:         %[[SLOT:.+]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    store i32 42, ptr %[[SLOT]], align 4
// LLVM:       }

// OGCG-LABEL: @f1
// OGCG:         %[[SLOT:.+]] = alloca i32, align 4
// OGCG-NEXT:    store i32 42, ptr %[[SLOT]], align 4
// OGCG:       }

void f2(void) {
  _Atomic(int) x;
  __c11_atomic_init(&x, 42);
}

// CIR-LABEL: @f2
// CIR:         %[[SLOT:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x"] {alignment = 4 : i64}
// CIR-NEXT:    %[[INIT:.+]] = cir.const #cir.int<42> : !s32i
// CIR-NEXT:    cir.store align(4) %[[INIT]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR:       }

// LLVM-LABEL: @f2
// LLVM:         %[[SLOT:.+]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    store i32 42, ptr %[[SLOT]], align 4
// LLVM:       }

// OGCG-LABEL: @f2
// OGCG:         %[[SLOT:.+]] = alloca i32, align 4
// OGCG-NEXT:    store i32 42, ptr %[[SLOT]], align 4
// OGCG:       }

void f3(_Atomic(int) *p) {
  *p = 42;
}

// CIR-LABEL: @f3
// CIR: cir.store align(4) atomic(seq_cst) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: @f3
// LLVM: store atomic i32 42, ptr %{{.+}} seq_cst, align 4

// OGCG-LABEL: @f3
// OGCG: store atomic i32 42, ptr %{{.+}} seq_cst, align 4

void f4(_Atomic(float) *p) {
  *p = 3.14;
}

// CIR-LABEL: @f4
// CIR: cir.store align(4) atomic(seq_cst) %{{.+}}, %{{.+}} : !cir.float, !cir.ptr<!cir.float>

// LLVM-LABEL: @f4
// LLVM: store atomic float 0x40091EB860000000, ptr %{{.+}} seq_cst, align 4

// OGCG-LABEL: @f4
// OGCG: store atomic float 0x40091EB860000000, ptr %{{.+}} seq_cst, align 4

void load(int *ptr) {
  int x;
  __atomic_load(ptr, &x, __ATOMIC_RELAXED);
  __atomic_load(ptr, &x, __ATOMIC_CONSUME);
  __atomic_load(ptr, &x, __ATOMIC_ACQUIRE);
  __atomic_load(ptr, &x, __ATOMIC_SEQ_CST);
}

// CIR-LABEL: @load
// CIR:   %{{.+}} = cir.load align(4) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(consume) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(seq_cst) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR: }

// LLVM-LABEL: @load
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} seq_cst, align 4
// LLVM: }

// OGCG-LABEL: @load
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} seq_cst, align 4
// OGCG: }

void load_n(int *ptr) {
  int a;
  a = __atomic_load_n(ptr, __ATOMIC_RELAXED);
  a = __atomic_load_n(ptr, __ATOMIC_CONSUME);
  a = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
  a = __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
}

// CIR-LABEL: @load_n
// CIR:   %{{.+}} = cir.load align(4) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(consume) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(seq_cst) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR: }

// LLVM-LABEL: @load_n
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} seq_cst, align 4
// LLVM: }

// OGCG-LABEL: @load_n
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} seq_cst, align 4
// OGCG: }

void c11_load(_Atomic(int) *ptr) {
  __c11_atomic_load(ptr, __ATOMIC_RELAXED);
  __c11_atomic_load(ptr, __ATOMIC_CONSUME);
  __c11_atomic_load(ptr, __ATOMIC_ACQUIRE);
  __c11_atomic_load(ptr, __ATOMIC_SEQ_CST);
}

// CIR-LABEL: @c11_load
// CIR:   %{{.+}} = cir.load align(4) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(consume) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) atomic(seq_cst) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR: }

// LLVM-LABEL: @c11_load
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// LLVM:   %{{.+}} = load atomic i32, ptr %{{.+}} seq_cst, align 4
// LLVM: }

// OGCG-LABEL: @c11_load
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} monotonic, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} acquire, align 4
// OGCG:   %{{.+}} = load atomic i32, ptr %{{.+}} seq_cst, align 4
// OGCG: }

void store(int *ptr, int x) {
  __atomic_store(ptr, &x, __ATOMIC_RELAXED);
  __atomic_store(ptr, &x, __ATOMIC_RELEASE);
  __atomic_store(ptr, &x, __ATOMIC_SEQ_CST);
}

// CIR-LABEL: @store
// CIR:   cir.store align(4) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) atomic(release) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) atomic(seq_cst) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR: }

// LLVM-LABEL: @store
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} release, align 4
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} seq_cst, align 4
// LLVM: }

// OGCG-LABEL: @store
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} release, align 4
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} seq_cst, align 4
// OGCG: }

void store_n(int *ptr, int x) {
  __atomic_store_n(ptr, x, __ATOMIC_RELAXED);
  __atomic_store_n(ptr, x, __ATOMIC_RELEASE);
  __atomic_store_n(ptr, x, __ATOMIC_SEQ_CST);
}

// CIR-LABEL: @store_n
// CIR:   cir.store align(4) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) atomic(release) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) atomic(seq_cst) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR: }

// LLVM-LABEL: @store_n
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} release, align 4
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} seq_cst, align 4
// LLVM: }

// OGCG-LABEL: @store_n
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} release, align 4
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} seq_cst, align 4
// OGCG: }

void c11_store(_Atomic(int) *ptr, int x) {
  __c11_atomic_store(ptr, x, __ATOMIC_RELAXED);
  __c11_atomic_store(ptr, x, __ATOMIC_RELEASE);
  __c11_atomic_store(ptr, x, __ATOMIC_SEQ_CST);
}

// CIR-LABEL: @c11_store
// CIR:   cir.store align(4) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) atomic(release) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) atomic(seq_cst) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR: }

// LLVM-LABEL: @c11_store
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} release, align 4
// LLVM:   store atomic i32 %{{.+}}, ptr %{{.+}} seq_cst, align 4
// LLVM: }

// OGCG-LABEL: @c11_store
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} monotonic, align 4
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} release, align 4
// OGCG:   store atomic i32 %{{.+}}, ptr %{{.+}} seq_cst, align 4
// OGCG: }

void c11_atomic_cmpxchg_strong(_Atomic(int) *ptr, int *expected, int desired) {
  // CIR-LABEL: @c11_atomic_cmpxchg_strong
  // LLVM-LABEL: @c11_atomic_cmpxchg_strong
  // OGCG-LABEL: @c11_atomic_cmpxchg_strong

  __c11_atomic_compare_exchange_strong(ptr, expected, desired,
                                       __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg success(seq_cst) failure(acquire) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR-NEXT:    %[[FAILED:.+]] = cir.unary(not, %[[SUCCESS]]) : !cir.bool, !cir.bool
  // CIR-NEXT:    cir.if %[[FAILED]] {
  // CIR-NEXT:      cir.store align(4) %[[OLD]], %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:    }
  // CIR-NEXT:    cir.store align(1) %[[SUCCESS]], %{{.+}} : !cir.bool, !cir.ptr<!cir.bool>

  // LLVM:         %[[RESULT:.+]] = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // LLVM-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // LLVM-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // LLVM-NEXT:    %[[FAILED:.+]] = xor i1 %[[SUCCESS]], true
  // LLVM-NEXT:    br i1 %[[FAILED]], label %[[LABEL_FAILED:.+]], label %[[LABEL_CONT:.+]]
  // LLVM:       [[LABEL_FAILED]]:
  // LLVM-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // LLVM-NEXT:    br label %[[LABEL_CONT]]
  // LLVM:       [[LABEL_CONT]]:
  // LLVM-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // LLVM-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1

  // OGCG:         %[[RESULT:.+]] = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // OGCG-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // OGCG-NEXT:    br i1 %[[SUCCESS]], label %[[LABEL_CONT:.+]], label %[[LABEL_FAILED:.+]]
  // OGCG:       [[LABEL_FAILED]]:
  // OGCG-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // OGCG-NEXT:    br label %[[LABEL_CONT]]
  // OGCG:       [[LABEL_CONT]]:
  // OGCG-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // OGCG-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1
}

void c11_atomic_cmpxchg_weak(_Atomic(int) *ptr, int *expected, int desired) {
  // CIR-LABEL: @c11_atomic_cmpxchg_weak
  // LLVM-LABEL: @c11_atomic_cmpxchg_weak
  // OGCG-LABEL: @c11_atomic_cmpxchg_weak

  __c11_atomic_compare_exchange_weak(ptr, expected, desired,
                                     __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR-NEXT:    %[[FAILED:.+]] = cir.unary(not, %[[SUCCESS]]) : !cir.bool, !cir.bool
  // CIR-NEXT:    cir.if %[[FAILED]] {
  // CIR-NEXT:      cir.store align(4) %[[OLD]], %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:    }
  // CIR-NEXT:    cir.store align(1) %[[SUCCESS]], %{{.+}} : !cir.bool, !cir.ptr<!cir.bool>

  // LLVM:         %[[RESULT:.+]] = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // LLVM-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // LLVM-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // LLVM-NEXT:    %[[FAILED:.+]] = xor i1 %[[SUCCESS]], true
  // LLVM-NEXT:    br i1 %[[FAILED]], label %[[LABEL_FAILED:.+]], label %[[LABEL_CONT:.+]]
  // LLVM:       [[LABEL_FAILED]]:
  // LLVM-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // LLVM-NEXT:    br label %[[LABEL_CONT]]
  // LLVM:       [[LABEL_CONT]]:
  // LLVM-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // LLVM-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1

  // OGCG:         %[[RESULT:.+]] = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // OGCG-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // OGCG-NEXT:    br i1 %[[SUCCESS]], label %[[LABEL_CONT:.+]], label %[[LABEL_FAILED:.+]]
  // OGCG:       [[LABEL_FAILED]]:
  // OGCG-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // OGCG-NEXT:    br label %[[LABEL_CONT]]
  // OGCG:       [[LABEL_CONT]]:
  // OGCG-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // OGCG-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1
}

void atomic_cmpxchg(int *ptr, int *expected, int *desired) {
  // CIR-LABEL: @atomic_cmpxchg
  // LLVM-LABEL: @atomic_cmpxchg
  // OGCG-LABEL: @atomic_cmpxchg

  __atomic_compare_exchange(ptr, expected, desired, /*weak=*/0, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg success(seq_cst) failure(acquire) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR-NEXT:    %[[FAILED:.+]] = cir.unary(not, %[[SUCCESS]]) : !cir.bool, !cir.bool
  // CIR-NEXT:    cir.if %[[FAILED]] {
  // CIR-NEXT:      cir.store align(4) %[[OLD]], %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:    }
  // CIR-NEXT:    cir.store align(1) %[[SUCCESS]], %{{.+}} : !cir.bool, !cir.ptr<!cir.bool>

  // LLVM:         %[[RESULT:.+]] = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // LLVM-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // LLVM-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // LLVM-NEXT:    %[[FAILED:.+]] = xor i1 %[[SUCCESS]], true
  // LLVM-NEXT:    br i1 %[[FAILED]], label %[[LABEL_FAILED:.+]], label %[[LABEL_CONT:.+]]
  // LLVM:       [[LABEL_FAILED]]:
  // LLVM-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // LLVM-NEXT:    br label %[[LABEL_CONT]]
  // LLVM:       [[LABEL_CONT]]:
  // LLVM-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // LLVM-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1

  // OGCG:         %[[RESULT:.+]] = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // OGCG-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // OGCG-NEXT:    br i1 %[[SUCCESS]], label %[[LABEL_CONT:.+]], label %[[LABEL_FAILED:.+]]
  // OGCG:       [[LABEL_FAILED]]:
  // OGCG-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // OGCG-NEXT:    br label %[[LABEL_CONT]]
  // OGCG:       [[LABEL_CONT]]:
  // OGCG-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // OGCG-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1

  __atomic_compare_exchange(ptr, expected, desired, /*weak=*/1, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR-NEXT:    %[[FAILED:.+]] = cir.unary(not, %[[SUCCESS]]) : !cir.bool, !cir.bool
  // CIR-NEXT:    cir.if %[[FAILED]] {
  // CIR-NEXT:      cir.store align(4) %[[OLD]], %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:    }
  // CIR-NEXT:    cir.store align(1) %[[SUCCESS]], %{{.+}} : !cir.bool, !cir.ptr<!cir.bool>

  // LLVM:         %[[RESULT:.+]] = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // LLVM-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // LLVM-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // LLVM-NEXT:    %[[FAILED:.+]] = xor i1 %[[SUCCESS]], true
  // LLVM-NEXT:    br i1 %[[FAILED]], label %[[LABEL_FAILED:.+]], label %[[LABEL_CONT:.+]]
  // LLVM:       [[LABEL_FAILED]]:
  // LLVM-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // LLVM-NEXT:    br label %[[LABEL_CONT]]
  // LLVM:       [[LABEL_CONT]]:
  // LLVM-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // LLVM-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1

  // OGCG:         %[[RESULT:.+]] = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // OGCG-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // OGCG-NEXT:    br i1 %[[SUCCESS]], label %[[LABEL_CONT:.+]], label %[[LABEL_FAILED:.+]]
  // OGCG:       [[LABEL_FAILED]]:
  // OGCG-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // OGCG-NEXT:    br label %[[LABEL_CONT]]
  // OGCG:       [[LABEL_CONT]]:
  // OGCG-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // OGCG-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1
}

void atomic_cmpxchg_n(int *ptr, int *expected, int desired) {
  // CIR-LABEL: @atomic_cmpxchg_n
  // LLVM-LABEL: @atomic_cmpxchg_n
  // OGCG-LABEL: @atomic_cmpxchg_n

  __atomic_compare_exchange_n(ptr, expected, desired, /*weak=*/0, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg success(seq_cst) failure(acquire) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR-NEXT:    %[[FAILED:.+]] = cir.unary(not, %[[SUCCESS]]) : !cir.bool, !cir.bool
  // CIR-NEXT:    cir.if %[[FAILED]] {
  // CIR-NEXT:      cir.store align(4) %[[OLD]], %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:    }
  // CIR-NEXT:    cir.store align(1) %[[SUCCESS]], %{{.+}} : !cir.bool, !cir.ptr<!cir.bool>

  // LLVM:         %[[RESULT:.+]] = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // LLVM-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // LLVM-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // LLVM-NEXT:    %[[FAILED:.+]] = xor i1 %[[SUCCESS]], true
  // LLVM-NEXT:    br i1 %[[FAILED]], label %[[LABEL_FAILED:.+]], label %[[LABEL_CONT:.+]]
  // LLVM:       [[LABEL_FAILED]]:
  // LLVM-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // LLVM-NEXT:    br label %[[LABEL_CONT]]
  // LLVM:       [[LABEL_CONT]]:
  // LLVM-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // LLVM-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1

  // OGCG:         %[[RESULT:.+]] = cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // OGCG-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // OGCG-NEXT:    br i1 %[[SUCCESS]], label %[[LABEL_CONT:.+]], label %[[LABEL_FAILED:.+]]
  // OGCG:       [[LABEL_FAILED]]:
  // OGCG-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // OGCG-NEXT:    br label %[[LABEL_CONT]]
  // OGCG:       [[LABEL_CONT]]:
  // OGCG-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // OGCG-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1

  __atomic_compare_exchange_n(ptr, expected, desired, /*weak=*/1, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
  // CIR-NEXT:    %[[FAILED:.+]] = cir.unary(not, %[[SUCCESS]]) : !cir.bool, !cir.bool
  // CIR-NEXT:    cir.if %[[FAILED]] {
  // CIR-NEXT:      cir.store align(4) %[[OLD]], %{{.+}} : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:    }
  // CIR-NEXT:    cir.store align(1) %[[SUCCESS]], %{{.+}} : !cir.bool, !cir.ptr<!cir.bool>

  // LLVM:         %[[RESULT:.+]] = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // LLVM-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // LLVM-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // LLVM-NEXT:    %[[FAILED:.+]] = xor i1 %[[SUCCESS]], true
  // LLVM-NEXT:    br i1 %[[FAILED]], label %[[LABEL_FAILED:.+]], label %[[LABEL_CONT:.+]]
  // LLVM:       [[LABEL_FAILED]]:
  // LLVM-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // LLVM-NEXT:    br label %[[LABEL_CONT]]
  // LLVM:       [[LABEL_CONT]]:
  // LLVM-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // LLVM-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1

  // OGCG:         %[[RESULT:.+]] = cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst acquire, align 4
  // OGCG-NEXT:    %[[OLD:.+]] = extractvalue { i32, i1 } %[[RESULT]], 0
  // OGCG-NEXT:    %[[SUCCESS:.+]] = extractvalue { i32, i1 } %[[RESULT]], 1
  // OGCG-NEXT:    br i1 %[[SUCCESS]], label %[[LABEL_CONT:.+]], label %[[LABEL_FAILED:.+]]
  // OGCG:       [[LABEL_FAILED]]:
  // OGCG-NEXT:    store i32 %[[OLD]], ptr %{{.+}}, align 4
  // OGCG-NEXT:    br label %[[LABEL_CONT]]
  // OGCG:       [[LABEL_CONT]]:
  // OGCG-NEXT:    %[[SUCCESS_2:.+]] = zext i1 %[[SUCCESS]] to i8
  // OGCG-NEXT:    store i8 %[[SUCCESS_2]], ptr %{{.+}}, align 1
}

void c11_atomic_exchange(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_exchange
  // LLVM-LABEL: @c11_atomic_exchange
  // OGCG-LABEL: @c11_atomic_exchange

  __c11_atomic_exchange(ptr, value, __ATOMIC_RELAXED);
  __c11_atomic_exchange(ptr, value, __ATOMIC_CONSUME);
  __c11_atomic_exchange(ptr, value, __ATOMIC_ACQUIRE);
  __c11_atomic_exchange(ptr, value, __ATOMIC_RELEASE);
  __c11_atomic_exchange(ptr, value, __ATOMIC_ACQ_REL);
  __c11_atomic_exchange(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.xchg relaxed %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg consume %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg acquire %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg release %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg acq_rel %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} release, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acq_rel, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4

  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} release, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acq_rel, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
}

void atomic_exchange(int *ptr, int *value, int *old) {
  // CIR-LABEL: @atomic_exchange
  // LLVM-LABEL: @atomic_exchange
  // OGCG-LABEL: @atomic_exchange

  __atomic_exchange(ptr, value, old, __ATOMIC_RELAXED);
  __atomic_exchange(ptr, value, old, __ATOMIC_CONSUME);
  __atomic_exchange(ptr, value, old, __ATOMIC_ACQUIRE);
  __atomic_exchange(ptr, value, old, __ATOMIC_RELEASE);
  __atomic_exchange(ptr, value, old, __ATOMIC_ACQ_REL);
  __atomic_exchange(ptr, value, old, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.xchg relaxed %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg consume %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg acquire %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg release %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg acq_rel %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} release, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acq_rel, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4

  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} release, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acq_rel, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
}

void atomic_exchange_n(int *ptr, int value) {
  // CIR-LABEL: @atomic_exchange_n
  // LLVM-LABEL: @atomic_exchange_n
  // OGCG-LABEL: @atomic_exchange_n

  __atomic_exchange_n(ptr, value, __ATOMIC_RELAXED);
  __atomic_exchange_n(ptr, value, __ATOMIC_CONSUME);
  __atomic_exchange_n(ptr, value, __ATOMIC_ACQUIRE);
  __atomic_exchange_n(ptr, value, __ATOMIC_RELEASE);
  __atomic_exchange_n(ptr, value, __ATOMIC_ACQ_REL);
  __atomic_exchange_n(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.xchg relaxed %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg consume %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg acquire %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg release %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg acq_rel %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: %{{.+}} = cir.atomic.xchg seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} release, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acq_rel, align 4
  // LLVM: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4

  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} monotonic, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acquire, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} release, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} acq_rel, align 4
  // OGCG: %{{.+}} = atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
}

void test_and_set(void *p) {
  // CIR-LABEL: @test_and_set
  // LLVM-LABEL: @test_and_set
  // OGCG-LABEL: @test_and_set

  __atomic_test_and_set(p, __ATOMIC_SEQ_CST);
  // CIR:      %[[VOID_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR-NEXT: %[[PTR:.+]] = cir.cast bitcast %[[VOID_PTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR-NEXT: %[[RES:.+]] = cir.atomic.test_and_set seq_cst %[[PTR]] : !cir.ptr<!s8i> -> !cir.bool
  // CIR-NEXT: cir.store align(1) %[[RES]], %{{.+}} : !cir.bool, !cir.ptr<!cir.bool>

  // LLVM:      %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i8 1 seq_cst, align 1
  // LLVM-NEXT: %{{.+}} = icmp ne i8 %[[RES]], 0

  // OGCG:      %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // OGCG-NEXT: %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i8 1 seq_cst, align 1
  // OGCG-NEXT: %{{.+}} = icmp ne i8 %[[RES]], 0
}

void test_and_set_volatile(volatile void *p) {
  // CIR-LABEL: @test_and_set_volatile
  // LLVM-LABEL: @test_and_set_volatile
  // OGCG-LABEL: @test_and_set_volatile

  __atomic_test_and_set(p, __ATOMIC_SEQ_CST);
  // CIR:      %[[VOID_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR-NEXT: %[[PTR:.+]] = cir.cast bitcast %[[VOID_PTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR-NEXT: %[[RES:.+]] = cir.atomic.test_and_set seq_cst %[[PTR]] volatile : !cir.ptr<!s8i> -> !cir.bool
  // CIR-NEXT: cir.store align(1) %[[RES]], %{{.+}} : !cir.bool, !cir.ptr<!cir.bool>

  // LLVM:      %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT: %[[RES:.+]] = atomicrmw volatile xchg ptr %[[PTR]], i8 1 seq_cst, align 1
  // LLVM-NEXT: %{{.+}} = icmp ne i8 %[[RES]], 0

  // OGCG:      %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // OGCG-NEXT: %[[RES:.+]] = atomicrmw volatile xchg ptr %[[PTR]], i8 1 seq_cst, align 1
  // OGCG-NEXT: %{{.+}} = icmp ne i8 %[[RES]], 0
}

void clear(void *p) {
  // CIR-LABEL: @clear
  // LLVM-LABEL: @clear
  // OGCG-LABEL: @clear

  __atomic_clear(p, __ATOMIC_SEQ_CST);
  // CIR:      %[[VOID_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR-NEXT: %[[PTR:.+]] = cir.cast bitcast %[[VOID_PTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR:      cir.atomic.clear seq_cst %[[PTR]] : !cir.ptr<!s8i>

  // LLVM: store atomic i8 0, ptr %{{.+}} seq_cst, align 1

  // OGCG: store atomic i8 0, ptr %{{.+}} seq_cst, align 1
}

void clear_volatile(volatile void *p) {
  // CIR-LABEL: @clear_volatile
  // LLVM-LABEL: @clear_volatile
  // OGCG-LABEL: @clear_volatile

  __atomic_clear(p, __ATOMIC_SEQ_CST);
  // CIR:      %[[VOID_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR-NEXT: %[[PTR:.+]] = cir.cast bitcast %[[VOID_PTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR:      cir.atomic.clear seq_cst %[[PTR]] volatile : !cir.ptr<!s8i>

  // LLVM: store atomic volatile i8 0, ptr %{{.+}} seq_cst, align 1

  // OGCG: store atomic volatile i8 0, ptr %{{.+}} seq_cst, align 1
}

int atomic_fetch_add(int *ptr, int value) {
  // CIR-LABEL: @atomic_fetch_add
  // LLVM-LABEL: @atomic_fetch_add
  // OGCG-LABEL: @atomic_fetch_add

  return __atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch add seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw add ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw add ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_add_fetch(int *ptr, int value) {
  // CIR-LABEL: @atomic_add_fetch
  // LLVM-LABEL: @atomic_add_fetch
  // OGCG-LABEL: @atomic_add_fetch

  return __atomic_add_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch add seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[OLD:.+]] = atomicrmw add ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[RES:.+]] = add i32 %[[OLD]], %[[VAL]]
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw add ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[RES:.+]] = add i32 %[[OLD]], %[[VAL]]
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int c11_atomic_fetch_add(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_fetch_add
  // LLVM-LABEL: @c11_atomic_fetch_add
  // OGCG-LABEL: @c11_atomic_fetch_add

  return __c11_atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch add seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw add ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw add ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_fetch_sub(int *ptr, int value) {
  // CIR-LABEL: @atomic_fetch_sub
  // LLVM-LABEL: @atomic_fetch_sub
  // OGCG-LABEL: @atomic_fetch_sub

  return __atomic_fetch_sub(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch sub seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw sub ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw sub ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_sub_fetch(int *ptr, int value) {
  // CIR-LABEL: @atomic_sub_fetch
  // LLVM-LABEL: @atomic_sub_fetch
  // OGCG-LABEL: @atomic_sub_fetch

  return __atomic_sub_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch sub seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[OLD:.+]] = atomicrmw sub ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[RES:.+]] = sub i32 %[[OLD]], %[[VAL]]
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw sub ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[RES:.+]] = sub i32 %[[OLD]], %[[VAL]]
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int c11_atomic_fetch_sub(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_fetch_sub
  // LLVM-LABEL: @c11_atomic_fetch_sub
  // OGCG-LABEL: @c11_atomic_fetch_sub

  return __c11_atomic_fetch_sub(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch sub seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw sub ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw sub ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

float atomic_fetch_add_fp(float *ptr, float value) {
  // CIR-LABEL: @atomic_fetch_add_fp
  // LLVM-LABEL: @atomic_fetch_add_fp
  // OGCG-LABEL: @atomic_fetch_add_fp

  return __atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch add seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[RES:.+]] = atomicrmw fadd ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw fadd ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

float atomic_add_fetch_fp(float *ptr, float value) {
  // CIR-LABEL: @atomic_add_fetch_fp
  // LLVM-LABEL: @atomic_add_fetch_fp
  // OGCG-LABEL: @atomic_add_fetch_fp

  return __atomic_add_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch add seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[OLD:.+]] = atomicrmw fadd ptr %{{.+}}, float %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[RES:.+]] = fadd float %[[OLD]], %[[VAL]]
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw fadd ptr %{{.+}}, float %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[RES:.+]] = fadd float %[[OLD]], %[[VAL]]
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

float c11_atomic_fetch_sub_fp(_Atomic(float) *ptr, float value) {
  // CIR-LABEL: @c11_atomic_fetch_sub_fp
  // LLVM-LABEL: @c11_atomic_fetch_sub_fp
  // OGCG-LABEL: @c11_atomic_fetch_sub_fp

  return __c11_atomic_fetch_sub(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch sub seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[RES:.+]] = atomicrmw fsub ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw fsub ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

int atomic_fetch_min(int *ptr, int value) {
  // CIR-LABEL: @atomic_fetch_min
  // LLVM-LABEL: @atomic_fetch_min
  // OGCG-LABEL: @atomic_fetch_min

  return __atomic_fetch_min(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch min seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw min ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw min ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_min_fetch(int *ptr, int value) {
  // CIR-LABEL: @atomic_min_fetch
  // LLVM-LABEL: @atomic_min_fetch
  // OGCG-LABEL: @atomic_min_fetch

  return __atomic_min_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch min seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[OLD:.+]] = atomicrmw min ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[OLD_LESS:.+]] = icmp slt i32 %[[OLD]], %[[VAL]]
  // LLVM-NEXT: %[[RES:.+]] = select i1 %[[OLD_LESS]], i32 %[[OLD]], i32 %[[VAL]]
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw min ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[OLD_LESS:.+]] = icmp slt i32 %[[OLD]], %[[VAL]]
  // OGCG-NEXT: %[[RES:.+]] = select i1 %[[OLD_LESS]], i32 %[[OLD]], i32 %[[VAL]]
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int c11_atomic_fetch_min(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_fetch_min
  // LLVM-LABEL: @c11_atomic_fetch_min
  // OGCG-LABEL: @c11_atomic_fetch_min

  return __c11_atomic_fetch_min(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch min seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw min ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw min ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

float atomic_fetch_min_fp(float *ptr, float value) {
  // CIR-LABEL: @atomic_fetch_min_fp
  // LLVM-LABEL: @atomic_fetch_min_fp
  // OGCG-LABEL: @atomic_fetch_min_fp

  return __atomic_fetch_min(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch min seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[RES:.+]] = atomicrmw fmin ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw fmin ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

float atomic_min_fetch_fp(float *ptr, float value) {
  // CIR-LABEL: @atomic_min_fetch_fp
  // LLVM-LABEL: @atomic_min_fetch_fp
  // OGCG-LABEL: @atomic_min_fetch_fp

  return __atomic_min_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch min seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[OLD:.+]] = atomicrmw fmin ptr %{{.+}}, float %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[RES:.+]] = call float @llvm.minnum.f32(float %[[OLD]], float %[[VAL]])
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw fmin ptr %{{.+}}, float %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[RES:.+]] = call float @llvm.minnum.f32(float %[[OLD]], float %[[VAL]])
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

float c11_atomic_fetch_min_fp(_Atomic(float) *ptr, float value) {
  // CIR-LABEL: @c11_atomic_fetch_min_fp
  // LLVM-LABEL: @c11_atomic_fetch_min_fp
  // OGCG-LABEL: @c11_atomic_fetch_min_fp

  return __c11_atomic_fetch_min(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch min seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[RES:.+]] = atomicrmw fmin ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw fmin ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

int atomic_fetch_max(int *ptr, int value) {
  // CIR-LABEL: @atomic_fetch_max
  // LLVM-LABEL: @atomic_fetch_max
  // OGCG-LABEL: @atomic_fetch_max

  return __atomic_fetch_max(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch max seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw max ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw max ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_max_fetch(int *ptr, int value) {
  // CIR-LABEL: @atomic_max_fetch
  // LLVM-LABEL: @atomic_max_fetch
  // OGCG-LABEL: @atomic_max_fetch

  return __atomic_max_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch max seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[OLD:.+]] = atomicrmw max ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[OLD_GREATER:.+]] = icmp sgt i32 %[[OLD]], %[[VAL]]
  // LLVM-NEXT: %[[RES:.+]] = select i1 %[[OLD_GREATER]], i32 %[[OLD]], i32 %[[VAL]]
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw max ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[OLD_GREATER:.+]] = icmp sgt i32 %[[OLD]], %[[VAL]]
  // OGCG-NEXT: %[[RES:.+]] = select i1 %[[OLD_GREATER]], i32 %[[OLD]], i32 %[[VAL]]
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int c11_atomic_fetch_max(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_fetch_max
  // LLVM-LABEL: @c11_atomic_fetch_max
  // OGCG-LABEL: @c11_atomic_fetch_max

  return __c11_atomic_fetch_max(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch max seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw max ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw max ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

float atomic_fetch_max_fp(float *ptr, float value) {
  // CIR-LABEL: @atomic_fetch_max_fp
  // LLVM-LABEL: @atomic_fetch_max_fp
  // OGCG-LABEL: @atomic_fetch_max_fp

  return __atomic_fetch_max(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch max seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[RES:.+]] = atomicrmw fmax ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw fmax ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

float atomic_max_fetch_fp(float *ptr, float value) {
  // CIR-LABEL: @atomic_max_fetch_fp
  // LLVM-LABEL: @atomic_max_fetch_fp
  // OGCG-LABEL: @atomic_max_fetch_fp

  return __atomic_max_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch max seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[OLD:.+]] = atomicrmw fmax ptr %{{.+}}, float %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[RES:.+]] = call float @llvm.maxnum.f32(float %[[OLD]], float %[[VAL]])
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw fmax ptr %{{.+}}, float %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[RES:.+]] = call float @llvm.maxnum.f32(float %[[OLD]], float %[[VAL]])
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

float c11_atomic_fetch_max_fp(_Atomic(float) *ptr, float value) {
  // CIR-LABEL: @c11_atomic_fetch_max_fp
  // LLVM-LABEL: @c11_atomic_fetch_max_fp
  // OGCG-LABEL: @c11_atomic_fetch_max_fp

  return __c11_atomic_fetch_max(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch max seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float

  // LLVM:      %[[RES:.+]] = atomicrmw fmax ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store float %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw fmax ptr %{{.+}}, float %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store float %[[RES]], ptr %{{.+}}, align 4
}

int atomic_fetch_and(int *ptr, int value) {
  // CIR-LABEL: @atomic_fetch_and
  // LLVM-LABEL: @atomic_fetch_and
  // OGCG-LABEL: @atomic_fetch_and

  return __atomic_fetch_and(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch and seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw and ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw and ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_and_fetch(int *ptr, int value) {
  // CIR-LABEL: @atomic_and_fetch
  // LLVM-LABEL: @atomic_and_fetch
  // OGCG-LABEL: @atomic_and_fetch

  return __atomic_and_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch and seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[OLD:.+]] = atomicrmw and ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[RES:.+]] = and i32 %[[OLD]], %[[VAL]]
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw and ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[RES:.+]] = and i32 %[[OLD]], %[[VAL]]
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int c11_atomic_fetch_and(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_fetch_and
  // LLVM-LABEL: @c11_atomic_fetch_and
  // OGCG-LABEL: @c11_atomic_fetch_and

  return __c11_atomic_fetch_and(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch and seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw and ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw and ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_fetch_or(int *ptr, int value) {
  // CIR-LABEL: @atomic_fetch_or
  // LLVM-LABEL: @atomic_fetch_or
  // OGCG-LABEL: @atomic_fetch_or

  return __atomic_fetch_or(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch or seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw or ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw or ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_or_fetch(int *ptr, int value) {
  // CIR-LABEL: @atomic_or_fetch
  // LLVM-LABEL: @atomic_or_fetch
  // OGCG-LABEL: @atomic_or_fetch

  return __atomic_or_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch or seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[OLD:.+]] = atomicrmw or ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[RES:.+]] = or i32 %[[OLD]], %[[VAL]]
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw or ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[RES:.+]] = or i32 %[[OLD]], %[[VAL]]
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int c11_atomic_fetch_or(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_fetch_or
  // LLVM-LABEL: @c11_atomic_fetch_or
  // OGCG-LABEL: @c11_atomic_fetch_or

  return __c11_atomic_fetch_or(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch or seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw or ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw or ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_fetch_xor(int *ptr, int value) {
  // CIR-LABEL: @atomic_fetch_xor
  // LLVM-LABEL: @atomic_fetch_xor
  // OGCG-LABEL: @atomic_fetch_xor

  return __atomic_fetch_xor(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch xor seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw xor ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw xor ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_xor_fetch(int *ptr, int value) {
  // CIR-LABEL: @atomic_xor_fetch
  // LLVM-LABEL: @atomic_xor_fetch
  // OGCG-LABEL: @atomic_xor_fetch

  return __atomic_xor_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch xor seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[OLD:.+]] = atomicrmw xor ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[RES:.+]] = xor i32 %[[OLD]], %[[VAL]]
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw xor ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[RES:.+]] = xor i32 %[[OLD]], %[[VAL]]
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int c11_atomic_fetch_xor(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_fetch_xor
  // LLVM-LABEL: @c11_atomic_fetch_xor
  // OGCG-LABEL: @c11_atomic_fetch_xor

  return __c11_atomic_fetch_xor(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch xor seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw xor ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw xor ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_fetch_nand(int *ptr, int value) {
  // CIR-LABEL: @atomic_fetch_nand
  // LLVM-LABEL: @atomic_fetch_nand
  // OGCG-LABEL: @atomic_fetch_nand

  return __atomic_fetch_nand(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch nand seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw nand ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw nand ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int atomic_nand_fetch(int *ptr, int value) {
  // CIR-LABEL: @atomic_nand_fetch
  // LLVM-LABEL: @atomic_nand_fetch
  // OGCG-LABEL: @atomic_nand_fetch

  return __atomic_nand_fetch(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch nand seq_cst %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[OLD:.+]] = atomicrmw nand ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // LLVM-NEXT: %[[TMP:.+]] = and i32 %[[OLD]], %[[VAL]]
  // LLVM-NEXT: %[[RES:.+]] = xor i32 %[[TMP]], -1
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[OLD:.+]] = atomicrmw nand ptr %{{.+}}, i32 %[[VAL:.+]] seq_cst, align 4
  // OGCG-NEXT: %[[TMP:.+]] = and i32 %[[OLD]], %[[VAL]]
  // OGCG-NEXT: %[[RES:.+]] = xor i32 %[[TMP]], -1
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}

int c11_atomic_fetch_nand(_Atomic(int) *ptr, int value) {
  // CIR-LABEL: @c11_atomic_fetch_nand
  // LLVM-LABEL: @c11_atomic_fetch_nand
  // OGCG-LABEL: @c11_atomic_fetch_nand

  return __c11_atomic_fetch_nand(ptr, value, __ATOMIC_SEQ_CST);
  // CIR: %{{.+}} = cir.atomic.fetch nand seq_cst fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i

  // LLVM:      %[[RES:.+]] = atomicrmw nand ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // LLVM-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4

  // OGCG:      %[[RES:.+]] = atomicrmw nand ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
  // OGCG-NEXT: store i32 %[[RES]], ptr %{{.+}}, align 4
}
