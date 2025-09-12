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
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg(%{{.+}} : !cir.ptr<!s32i>, %{{.+}} : !s32i, %{{.+}} : !s32i, success = seq_cst, failure = acquire) align(4) : (!s32i, !cir.bool)
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
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg(%{{.+}} : !cir.ptr<!s32i>, %{{.+}} : !s32i, %{{.+}} : !s32i, success = seq_cst, failure = acquire) align(4) weak : (!s32i, !cir.bool)
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
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg(%{{.+}} : !cir.ptr<!s32i>, %{{.+}} : !s32i, %{{.+}} : !s32i, success = seq_cst, failure = acquire) align(4) : (!s32i, !cir.bool)
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
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg(%{{.+}} : !cir.ptr<!s32i>, %{{.+}} : !s32i, %{{.+}} : !s32i, success = seq_cst, failure = acquire) align(4) weak : (!s32i, !cir.bool)
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
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg(%{{.+}} : !cir.ptr<!s32i>, %{{.+}} : !s32i, %{{.+}} : !s32i, success = seq_cst, failure = acquire) align(4) : (!s32i, !cir.bool)
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
  // CIR:         %[[OLD:.+]], %[[SUCCESS:.+]] = cir.atomic.cmpxchg(%{{.+}} : !cir.ptr<!s32i>, %{{.+}} : !s32i, %{{.+}} : !s32i, success = seq_cst, failure = acquire) align(4) weak : (!s32i, !cir.bool)
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
