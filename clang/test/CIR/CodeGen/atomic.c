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
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(seq_cst) %{{.+}} : !cir.ptr<!s32i>, !s32i
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
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(seq_cst) %{{.+}} : !cir.ptr<!s32i>, !s32i
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
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(relaxed) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(acquire) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.+}} = cir.load align(4) syncscope(system) atomic(seq_cst) %{{.+}} : !cir.ptr<!s32i>, !s32i
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

struct Pair {
  int x;
  int y;
};

void c11_load_aggregate() {
  _Atomic(struct Pair) a;
  __c11_atomic_load(&a, __ATOMIC_RELAXED);
  __c11_atomic_load(&a, __ATOMIC_ACQUIRE);
  __c11_atomic_load(&a, __ATOMIC_SEQ_CST);
}

// CIR-LABEL: @c11_load_aggregate
// CIR: %{{.*}} = cir.load {{.*}} syncscope(system) atomic(relaxed) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR: %{{.*}} = cir.load {{.*}} syncscope(system) atomic(acquire) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR: %{{.*}} = cir.load {{.*}} syncscope(system) atomic(seq_cst) %{{.*}} : !cir.ptr<!u64i>, !u64i

// LLVM-LABEL: @c11_load_aggregate
// LLVM: %{{.*}} = load atomic i64, ptr %{{.*}} monotonic, align 8
// LLVM: %{{.*}} = load atomic i64, ptr %{{.*}} acquire, align 8
// LLVM: %{{.*}} = load atomic i64, ptr %{{.*}} seq_cst, align 8

// OGCG-LABEL: @c11_load_aggregate
// OGCG: %{{.*}} = load atomic i64, ptr %{{.*}} monotonic, align 8
// OGCG: %{{.*}} = load atomic i64, ptr %{{.*}} acquire, align 8
// OGCG: %{{.*}} = load atomic i64, ptr %{{.*}} seq_cst, align 8

void store(int *ptr, int x) {
  __atomic_store(ptr, &x, __ATOMIC_RELAXED);
  __atomic_store(ptr, &x, __ATOMIC_RELEASE);
  __atomic_store(ptr, &x, __ATOMIC_SEQ_CST);
}

// CIR-LABEL: @store
// CIR:   cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) syncscope(system) atomic(release) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) syncscope(system) atomic(seq_cst) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
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
// CIR:   cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) syncscope(system) atomic(release) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) syncscope(system) atomic(seq_cst) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
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
// CIR:   cir.store align(4) syncscope(system) atomic(relaxed) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) syncscope(system) atomic(release) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) syncscope(system) atomic(seq_cst) %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>
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
  // CIR: %{{.+}} = cir.atomic.xchg acquire %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
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
  // CIR: %{{.+}} = cir.atomic.xchg acquire %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
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
  // CIR: %{{.+}} = cir.atomic.xchg acquire %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
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
  // CIR:      %[[RES:.+]] = cir.atomic.test_and_set seq_cst %[[PTR]] : !cir.ptr<!s8i> -> !cir.bool
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
  // CIR:      %[[RES:.+]] = cir.atomic.test_and_set seq_cst %[[PTR]] volatile : !cir.ptr<!s8i> -> !cir.bool
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

// CHECK-LABEL: @test_op_and_fetch
// LLVM-LABEL: @test_op_and_fetch
void test_op_and_fetch() {
  int *ptr;
  signed char sc;
  unsigned char uc;
  signed short ss;
  unsigned short us;
  signed int si;
  unsigned int ui;
  signed long long sll;
  unsigned long long ull;

  // CIR: [[RES0:%.*]] = cir.load align(8) {{%.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: [[VAL0:%.*]] = cir.cast bitcast {{%.*}} : !cir.ptr<!cir.ptr<!s32i>> -> !cir.ptr<!s64i>
  // CIR: [[VAL1:%.*]] = cir.cast ptr_to_int {{%.*}} : !cir.ptr<!s32i> -> !s64i
  // CIR: [[RES1:%.*]] = cir.atomic.fetch add seq_cst fetch_first [[VAL0]], [[VAL1]] : (!cir.ptr<!s64i>, !s64i) -> !s64i
  // CIR: [[RES2:%.*]] = cir.binop(add, [[RES1]], [[VAL1]]) : !s64i
  // CIR: [[RES3:%.*]] = cir.cast int_to_ptr [[RES2]] : !s64i -> !cir.ptr<!s32i>
  // LLVM:  [[VAL0:%.*]] = load ptr, ptr %{{.*}}, align 8
  // LLVM:  [[VAL1:%.*]] = ptrtoint ptr %{{.*}} to i64
  // LLVM:  [[RES0:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[VAL1]] seq_cst, align 8
  // LLVM:  [[RET0:%.*]] = add i64 [[RES0]], [[VAL1]]
  // LLVM:  [[RET1:%.*]] = inttoptr i64 [[RET0]] to ptr
  // LLVM:  store ptr [[RET1]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL0:%.*]] = load ptr, ptr %{{.*}}, align 8
  // OGCG:  [[VAL1:%.*]] = ptrtoint ptr %{{.*}} to i64
  // OGCG:  [[RES0:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[VAL1]] seq_cst, align 8
  // OGCG:  [[RET0:%.*]] = add i64 [[RES0]], [[VAL1]]
  // OGCG:  [[RET1:%.*]] = inttoptr i64 [[RET0]] to ptr
  // OGCG:  store ptr [[RET1]], ptr %{{.*}}, align 8
  ptr = __sync_add_and_fetch(&ptr, ptr);

  // CIR: [[VAL0:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s8i
  // CIR: [[RES0:%.*]] = cir.atomic.fetch add seq_cst fetch_first {{%.*}}, [[VAL0]] : (!cir.ptr<!s8i>, !s8i) -> !s8i
  // CIR: [[RET0:%.*]] = cir.binop(add, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw add ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = add i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES0:%.*]] = atomicrmw add ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // OGCG:  [[RET0:%.*]] = add i8 [[RES0]], [[VAL0]]
  // OGCG:  store i8 [[RET0]], ptr %{{.*}}, align 1
  sc = __sync_add_and_fetch(&sc, uc);

  // CIR: [[RES1:%.*]] = cir.atomic.fetch add seq_cst fetch_first {{%.*}}, [[VAL1:%.*]] : (!cir.ptr<!u8i>, !u8i) -> !u8i
  // CIR: [[RET1:%.*]] = cir.binop(add, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw add ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = add i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES1:%.*]] = atomicrmw add ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // OGCG:  [[RET1:%.*]] = add i8 [[RES1]], [[VAL1]]
  // OGCG:  store i8 [[RET1]], ptr %{{.*}}, align 1
  uc = __sync_add_and_fetch(&uc, uc);

  // CIR: [[VAL2:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s16i
  // CIR: [[RES2:%.*]] = cir.atomic.fetch add seq_cst fetch_first {{%.*}}, [[VAL2]] : (!cir.ptr<!s16i>, !s16i) -> !s16i
  // CIR: [[RET2:%.*]] = cir.binop(add, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw add ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = add i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr %{{.*}}, align 2
  // OGCG:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // OGCG:  [[RES2:%.*]] = atomicrmw add ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // OGCG:  [[RET2:%.*]] = add i16 [[RES2]], [[CONV2]]
  // OGCG:  store i16 [[RET2]], ptr %{{.*}}, align 2
  ss = __sync_add_and_fetch(&ss, uc);

  // CIR: [[VAL3:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u16i
  // CIR: [[RES3:%.*]] = cir.atomic.fetch add seq_cst fetch_first {{%.*}}, [[VAL3]] : (!cir.ptr<!u16i>, !u16i) -> !u16i
  // CIR: [[RET3:%.*]] = cir.binop(add, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw add ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = add i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr %{{.*}}
  // OGCG:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // OGCG:  [[RES3:%.*]] = atomicrmw add ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // OGCG:  [[RET3:%.*]] = add i16 [[RES3]], [[CONV3]]
  // OGCG:  store i16 [[RET3]], ptr %{{.*}}
  us = __sync_add_and_fetch(&us, uc);

  // CIR: [[VAL4:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s32i
  // CIR: [[RES4:%.*]] = cir.atomic.fetch add seq_cst fetch_first {{%.*}}, [[VAL4]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: [[RET4:%.*]] = cir.binop(add, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw add ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = add i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // OGCG:  [[RES4:%.*]] = atomicrmw add ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // OGCG:  [[RET4:%.*]] = add i32 [[RES4]], [[CONV4]]
  // OGCG:  store i32 [[RET4]], ptr %{{.*}}, align 4
  si = __sync_add_and_fetch(&si, uc);

  // CIR: [[VAL5:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u32i
  // CIR: [[RES5:%.*]] = cir.atomic.fetch add seq_cst fetch_first {{%.*}}, [[VAL5]] : (!cir.ptr<!u32i>, !u32i) -> !u32i
  // CIR: [[RET5:%.*]] = cir.binop(add, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw add ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = add i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // OGCG:  [[RES5:%.*]] = atomicrmw add ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // OGCG:  [[RET5:%.*]] = add i32 [[RES5]], [[CONV5]]
  // OGCG:  store i32 [[RET5]], ptr %{{.*}}, align 4
  ui = __sync_add_and_fetch(&ui, uc);

  // CIR: [[VAL6:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s64i
  // CIR: [[RES6:%.*]] = cir.atomic.fetch add seq_cst fetch_first {{%.*}}, [[VAL6]] : (!cir.ptr<!s64i>, !s64i) -> !s64i
  // CIR: [[RET6:%.*]] = cir.binop(add, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = add i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // OGCG:  [[RES6:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // OGCG:  [[RET6:%.*]] = add i64 [[RES6]], [[CONV6]]
  // OGCG:  store i64 [[RET6]], ptr %{{.*}}, align 8
  sll = __sync_add_and_fetch(&sll, uc);

  // CIR: [[VAL7:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u64i
  // CIR: [[RES7:%.*]] = cir.atomic.fetch add seq_cst fetch_first {{%.*}}, [[VAL7]] : (!cir.ptr<!u64i>, !u64i) -> !u64i
  // CIR: [[RET7:%.*]] = cir.binop(add, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = add i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // OGCG:  [[RES7:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // OGCG:  [[RET7:%.*]] = add i64 [[RES7]], [[CONV7]]
  // OGCG:  store i64 [[RET7]], ptr %{{.*}}, align 8
  ull = __sync_add_and_fetch(&ull, uc);

  // CIR: [[VAL0:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s8i
  // CIR: [[RES0:%.*]] = cir.atomic.fetch sub seq_cst fetch_first {{%.*}}, [[VAL0]] : (!cir.ptr<!s8i>, !s8i) -> !s8i
  // CIR: [[RET0:%.*]] = cir.binop(sub, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw sub ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = sub i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES0:%.*]] = atomicrmw sub ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // OGCG:  [[RET0:%.*]] = sub i8 [[RES0]], [[VAL0]]
  // OGCG:  store i8 [[RET0]], ptr %{{.*}}, align 1
  sc = __sync_sub_and_fetch(&sc, uc);

  // CIR: [[RES1:%.*]] = cir.atomic.fetch sub seq_cst fetch_first {{%.*}}, [[VAL1:%.*]] : (!cir.ptr<!u8i>, !u8i) -> !u8i
  // CIR: [[RET1:%.*]] = cir.binop(sub, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw sub ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = sub i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES1:%.*]] = atomicrmw sub ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // OGCG:  [[RET1:%.*]] = sub i8 [[RES1]], [[VAL1]]
  // OGCG:  store i8 [[RET1]], ptr %{{.*}}, align 1
  uc = __sync_sub_and_fetch(&uc, uc);

  // CIR: [[VAL2:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s16i
  // CIR: [[RES2:%.*]] = cir.atomic.fetch sub seq_cst fetch_first {{%.*}}, [[VAL2]] : (!cir.ptr<!s16i>, !s16i) -> !s16i
  // CIR: [[RET2:%.*]] = cir.binop(sub, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw sub ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = sub i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr %{{.*}}, align 2
  // OGCG:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // OGCG:  [[RES2:%.*]] = atomicrmw sub ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // OGCG:  [[RET2:%.*]] = sub i16 [[RES2]], [[CONV2]]
  // OGCG:  store i16 [[RET2]], ptr %{{.*}}, align 2
  ss = __sync_sub_and_fetch(&ss, uc);

  // CIR: [[VAL3:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u16i
  // CIR: [[RES3:%.*]] = cir.atomic.fetch sub seq_cst fetch_first {{%.*}}, [[VAL3]] : (!cir.ptr<!u16i>, !u16i) -> !u16i
  // CIR: [[RET3:%.*]] = cir.binop(sub, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw sub ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = sub i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr %{{.*}}
  // OGCG:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // OGCG:  [[RES3:%.*]] = atomicrmw sub ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // OGCG:  [[RET3:%.*]] = sub i16 [[RES3]], [[CONV3]]
  // OGCG:  store i16 [[RET3]], ptr %{{.*}}
  us = __sync_sub_and_fetch(&us, uc);

  // CIR: [[VAL4:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s32i
  // CIR: [[RES4:%.*]] = cir.atomic.fetch sub seq_cst fetch_first {{%.*}}, [[VAL4]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: [[RET4:%.*]] = cir.binop(sub, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw sub ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = sub i32 [[RES4]], [[CONV4]]
  // OGCG:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // OGCG:  [[RES4:%.*]] = atomicrmw sub ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // OGCG:  [[RET4:%.*]] = sub i32 [[RES4]], [[CONV4]]
  // OGCG:  store i32 [[RET4]], ptr %{{.*}}, align 4
  si = __sync_sub_and_fetch(&si, uc);

  // CIR: [[VAL5:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u32i
  // CIR: [[RES5:%.*]] = cir.atomic.fetch sub seq_cst fetch_first {{%.*}}, [[VAL5]] : (!cir.ptr<!u32i>, !u32i) -> !u32i
  // CIR: [[RET5:%.*]] = cir.binop(sub, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw sub ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = sub i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // OGCG:  [[RES5:%.*]] = atomicrmw sub ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // OGCG:  [[RET5:%.*]] = sub i32 [[RES5]], [[CONV5]]
  // OGCG:  store i32 [[RET5]], ptr %{{.*}}, align 4
  ui = __sync_sub_and_fetch(&ui, uc);

  // CIR: [[VAL6:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s64i
  // CIR: [[RES6:%.*]] = cir.atomic.fetch sub seq_cst fetch_first {{%.*}}, [[VAL6]] : (!cir.ptr<!s64i>, !s64i) -> !s64i
  // CIR: [[RET6:%.*]] = cir.binop(sub, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw sub ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = sub i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // OGCG:  [[RES6:%.*]] = atomicrmw sub ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // OGCG:  [[RET6:%.*]] = sub i64 [[RES6]], [[CONV6]]
  // OGCG:  store i64 [[RET6]], ptr %{{.*}}, align 8
  sll = __sync_sub_and_fetch(&sll, uc);

  // CIR: [[VAL7:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u64i
  // CIR: [[RES7:%.*]] = cir.atomic.fetch sub seq_cst fetch_first {{%.*}}, [[VAL7]] : (!cir.ptr<!u64i>, !u64i) -> !u64i
  // CIR: [[RET7:%.*]] = cir.binop(sub, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw sub ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = sub i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // OGCG:  [[RES7:%.*]] = atomicrmw sub ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // OGCG:  [[RET7:%.*]] = sub i64 [[RES7]], [[CONV7]]
  // OGCG:  store i64 [[RET7]], ptr %{{.*}}, align 8
  ull = __sync_sub_and_fetch(&ull, uc);

  // CIR: [[VAL0:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s8i
  // CIR: [[RES0:%.*]] = cir.atomic.fetch and seq_cst fetch_first {{%.*}}, [[VAL0]] : (!cir.ptr<!s8i>, !s8i) -> !s8i
  // CIR: [[RET0:%.*]] = cir.binop(and, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw and ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = and i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES0:%.*]] = atomicrmw and ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // OGCG:  [[RET0:%.*]] = and i8 [[RES0]], [[VAL0]]
  // OGCG:  store i8 [[RET0]], ptr %{{.*}}, align 1
  sc = __sync_and_and_fetch(&sc, uc);

  // CIR: [[RES1:%.*]] = cir.atomic.fetch and seq_cst fetch_first {{%.*}}, [[VAL1:%.*]] : (!cir.ptr<!u8i>, !u8i) -> !u8i
  // CIR: [[RET1:%.*]] = cir.binop(and, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw and ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = and i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES1:%.*]] = atomicrmw and ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // OGCG:  [[RET1:%.*]] = and i8 [[RES1]], [[VAL1]]
  // OGCG:  store i8 [[RET1]], ptr %{{.*}}, align 1
  uc = __sync_and_and_fetch(&uc, uc);

  // CIR: [[VAL2:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s16i
  // CIR: [[RES2:%.*]] = cir.atomic.fetch and seq_cst fetch_first {{%.*}}, [[VAL2]] : (!cir.ptr<!s16i>, !s16i) -> !s16i
  // CIR: [[RET2:%.*]] = cir.binop(and, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw and ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = and i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr %{{.*}}, align 2
  // OGCG:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // OGCG:  [[RES2:%.*]] = atomicrmw and ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // OGCG:  [[RET2:%.*]] = and i16 [[RES2]], [[CONV2]]
  // OGCG:  store i16 [[RET2]], ptr %{{.*}}, align 2
  ss = __sync_and_and_fetch(&ss, uc);

  // CIR: [[VAL3:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u16i
  // CIR: [[RES3:%.*]] = cir.atomic.fetch and seq_cst fetch_first {{%.*}}, [[VAL3]] : (!cir.ptr<!u16i>, !u16i) -> !u16i
  // CIR: [[RET3:%.*]] = cir.binop(and, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw and ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = and i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr %{{.*}}
  // OGCG:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // OGCG:  [[RES3:%.*]] = atomicrmw and ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // OGCG:  [[RET3:%.*]] = and i16 [[RES3]], [[CONV3]]
  // OGCG:  store i16 [[RET3]], ptr %{{.*}}
  us = __sync_and_and_fetch(&us, uc);

  // CIR: [[VAL4:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s32i
  // CIR: [[RES4:%.*]] = cir.atomic.fetch and seq_cst fetch_first {{%.*}}, [[VAL4]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: [[RET4:%.*]] = cir.binop(and, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw and ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = and i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // OGCG:  [[RES4:%.*]] = atomicrmw and ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // OGCG:  [[RET4:%.*]] = and i32 [[RES4]], [[CONV4]]
  // OGCG:  store i32 [[RET4]], ptr %{{.*}}, align 4
  si = __sync_and_and_fetch(&si, uc);

  // CIR: [[VAL5:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u32i
  // CIR: [[RES5:%.*]] = cir.atomic.fetch and seq_cst fetch_first {{%.*}}, [[VAL5]] : (!cir.ptr<!u32i>, !u32i) -> !u32i
  // CIR: [[RET5:%.*]] = cir.binop(and, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw and ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = and i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // OGCG:  [[RES5:%.*]] = atomicrmw and ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // OGCG:  [[RET5:%.*]] = and i32 [[RES5]], [[CONV5]]
  // OGCG:  store i32 [[RET5]], ptr %{{.*}}, align 4
  ui = __sync_and_and_fetch(&ui, uc);

  // CIR: [[VAL6:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s64i
  // CIR: [[RES6:%.*]] = cir.atomic.fetch and seq_cst fetch_first {{%.*}}, [[VAL6]] : (!cir.ptr<!s64i>, !s64i) -> !s64i
  // CIR: [[RET6:%.*]] = cir.binop(and, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw and ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = and i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // OGCG:  [[RES6:%.*]] = atomicrmw and ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // OGCG:  [[RET6:%.*]] = and i64 [[RES6]], [[CONV6]]
  // OGCG:  store i64 [[RET6]], ptr %{{.*}}, align 8
  sll = __sync_and_and_fetch(&sll, uc);

  // CIR: [[VAL7:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u64i
  // CIR: [[RES7:%.*]] = cir.atomic.fetch and seq_cst fetch_first {{%.*}}, [[VAL7]] : (!cir.ptr<!u64i>, !u64i) -> !u64i
  // CIR: [[RET7:%.*]] = cir.binop(and, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw and ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = and i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // OGCG:  [[RES7:%.*]] = atomicrmw and ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // OGCG:  [[RET7:%.*]] = and i64 [[RES7]], [[CONV7]]
  // OGCG:  store i64 [[RET7]], ptr %{{.*}}, align 8
  ull = __sync_and_and_fetch(&ull, uc);

  // CIR: [[VAL0:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s8i
  // CIR: [[RES0:%.*]] = cir.atomic.fetch or seq_cst fetch_first {{%.*}}, [[VAL0]] : (!cir.ptr<!s8i>, !s8i) -> !s8i
  // CIR: [[RET0:%.*]] = cir.binop(or, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw or ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = or i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES0:%.*]] = atomicrmw or ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // OGCG:  [[RET0:%.*]] = or i8 [[RES0]], [[VAL0]]
  // OGCG:  store i8 [[RET0]], ptr %{{.*}}, align 1
  sc = __sync_or_and_fetch(&sc, uc);

  // CIR: [[RES1:%.*]] = cir.atomic.fetch or seq_cst fetch_first {{%.*}}, [[VAL1:%.*]] : (!cir.ptr<!u8i>, !u8i) -> !u8i
  // CIR: [[RET1:%.*]] = cir.binop(or, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw or ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = or i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES1:%.*]] = atomicrmw or ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // OGCG:  [[RET1:%.*]] = or i8 [[RES1]], [[VAL1]]
  // OGCG:  store i8 [[RET1]], ptr %{{.*}}, align 1
  uc = __sync_or_and_fetch(&uc, uc);

  // CIR: [[VAL2:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s16i
  // CIR: [[RES2:%.*]] = cir.atomic.fetch or seq_cst fetch_first {{%.*}}, [[VAL2]] : (!cir.ptr<!s16i>, !s16i) -> !s16i
  // CIR: [[RET2:%.*]] = cir.binop(or, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw or ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = or i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr %{{.*}}, align 2
  // OGCG:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // OGCG:  [[RES2:%.*]] = atomicrmw or ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // OGCG:  [[RET2:%.*]] = or i16 [[RES2]], [[CONV2]]
  // OGCG:  store i16 [[RET2]], ptr %{{.*}}, align 2
  ss = __sync_or_and_fetch(&ss, uc);

  // CIR: [[VAL3:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u16i
  // CIR: [[RES3:%.*]] = cir.atomic.fetch or seq_cst fetch_first {{%.*}}, [[VAL3]] : (!cir.ptr<!u16i>, !u16i) -> !u16i
  // CIR: [[RET3:%.*]] = cir.binop(or, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw or ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = or i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr %{{.*}}
  // OGCG:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // OGCG:  [[RES3:%.*]] = atomicrmw or ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // OGCG:  [[RET3:%.*]] = or i16 [[RES3]], [[CONV3]]
  // OGCG:  store i16 [[RET3]], ptr %{{.*}}
  us = __sync_or_and_fetch(&us, uc);

  // CIR: [[VAL4:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s32i
  // CIR: [[RES4:%.*]] = cir.atomic.fetch or seq_cst fetch_first {{%.*}}, [[VAL4]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: [[RET4:%.*]] = cir.binop(or, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw or ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = or i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // OGCG:  [[RES4:%.*]] = atomicrmw or ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // OGCG:  [[RET4:%.*]] = or i32 [[RES4]], [[CONV4]]
  // OGCG:  store i32 [[RET4]], ptr %{{.*}}, align 4
  si = __sync_or_and_fetch(&si, uc);

  // CIR: [[VAL5:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u32i
  // CIR: [[RES5:%.*]] = cir.atomic.fetch or seq_cst fetch_first {{%.*}}, [[VAL5]] : (!cir.ptr<!u32i>, !u32i) -> !u32i
  // CIR: [[RET5:%.*]] = cir.binop(or, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw or ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = or i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // OGCG:  [[RES5:%.*]] = atomicrmw or ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // OGCG:  [[RET5:%.*]] = or i32 [[RES5]], [[CONV5]]
  // OGCG:  store i32 [[RET5]], ptr %{{.*}}, align 4
  ui = __sync_or_and_fetch(&ui, uc);

  // CIR: [[VAL6:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s64i
  // CIR: [[RES6:%.*]] = cir.atomic.fetch or seq_cst fetch_first {{%.*}}, [[VAL6]] : (!cir.ptr<!s64i>, !s64i) -> !s64i
  // CIR: [[RET6:%.*]] = cir.binop(or, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw or ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = or i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // OGCG:  [[RES6:%.*]] = atomicrmw or ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // OGCG:  [[RET6:%.*]] = or i64 [[RES6]], [[CONV6]]
  // OGCG:  store i64 [[RET6]], ptr %{{.*}}, align 8
  sll = __sync_or_and_fetch(&sll, uc);

  // CIR: [[VAL7:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u64i
  // CIR: [[RES7:%.*]] = cir.atomic.fetch or seq_cst fetch_first {{%.*}}, [[VAL7]] : (!cir.ptr<!u64i>, !u64i) -> !u64i
  // CIR: [[RET7:%.*]] = cir.binop(or, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw or ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = or i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // OGCG:  [[RES7:%.*]] = atomicrmw or ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // OGCG:  [[RET7:%.*]] = or i64 [[RES7]], [[CONV7]]
  // OGCG:  store i64 [[RET7]], ptr %{{.*}}, align 8
  ull = __sync_or_and_fetch(&ull, uc);

  // CIR: [[VAL0:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s8i
  // CIR: [[RES0:%.*]] = cir.atomic.fetch xor seq_cst fetch_first {{%.*}}, [[VAL0]] : (!cir.ptr<!s8i>, !s8i) -> !s8i
  // CIR: [[RET0:%.*]] = cir.binop(xor, [[RES0]], [[VAL0]]) : !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw xor ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[RET0:%.*]] = xor i8 [[RES0]], [[VAL0]]
  // LLVM:  store i8 [[RET0]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES0:%.*]] = atomicrmw xor ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // OGCG:  [[RET0:%.*]] = xor i8 [[RES0]], [[VAL0]]
  // OGCG:  store i8 [[RET0]], ptr %{{.*}}, align 1
  sc = __sync_xor_and_fetch(&sc, uc);

  // CIR: [[RES1:%.*]] = cir.atomic.fetch xor seq_cst fetch_first {{%.*}}, [[VAL1:%.*]] : (!cir.ptr<!u8i>, !u8i) -> !u8i
  // CIR: [[RET1:%.*]] = cir.binop(xor, [[RES1]], [[VAL1]]) : !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw xor ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[RET1:%.*]] = xor i8 [[RES1]], [[VAL1]]
  // LLVM:  store i8 [[RET1]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES1:%.*]] = atomicrmw xor ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // OGCG:  [[RET1:%.*]] = xor i8 [[RES1]], [[VAL1]]
  // OGCG:  store i8 [[RET1]], ptr %{{.*}}, align 1
  uc = __sync_xor_and_fetch(&uc, uc);

  // CIR: [[VAL2:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s16i
  // CIR: [[RES2:%.*]] = cir.atomic.fetch xor seq_cst fetch_first {{%.*}}, [[VAL2]] : (!cir.ptr<!s16i>, !s16i) -> !s16i
  // CIR: [[RET2:%.*]] = cir.binop(xor, [[RES2]], [[VAL2]]) : !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw xor ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[RET2:%.*]] = xor i16 [[RES2]], [[CONV2]]
  // LLVM:  store i16 [[RET2]], ptr %{{.*}}, align 2
  // OGCG:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // OGCG:  [[RES2:%.*]] = atomicrmw xor ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // OGCG:  [[RET2:%.*]] = xor i16 [[RES2]], [[CONV2]]
  // OGCG:  store i16 [[RET2]], ptr %{{.*}}, align 2
  ss = __sync_xor_and_fetch(&ss, uc);

  // CIR: [[VAL3:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u16i
  // CIR: [[RES3:%.*]] = cir.atomic.fetch xor seq_cst fetch_first {{%.*}}, [[VAL3]] : (!cir.ptr<!u16i>, !u16i) -> !u16i
  // CIR: [[RET3:%.*]] = cir.binop(xor, [[RES3]], [[VAL3]]) : !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw xor ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[RET3:%.*]] = xor i16 [[RES3]], [[CONV3]]
  // LLVM:  store i16 [[RET3]], ptr %{{.*}}
  // OGCG:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // OGCG:  [[RES3:%.*]] = atomicrmw xor ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // OGCG:  [[RET3:%.*]] = xor i16 [[RES3]], [[CONV3]]
  // OGCG:  store i16 [[RET3]], ptr %{{.*}}
  us = __sync_xor_and_fetch(&us, uc);

  // CIR: [[VAL4:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s32i
  // CIR: [[RES4:%.*]] = cir.atomic.fetch xor seq_cst fetch_first {{%.*}}, [[VAL4]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: [[RET4:%.*]] = cir.binop(xor, [[RES4]], [[VAL4]]) : !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw xor ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[RET4:%.*]] = xor i32 [[RES4]], [[CONV4]]
  // LLVM:  store i32 [[RET4]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // OGCG:  [[RES4:%.*]] = atomicrmw xor ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // OGCG:  [[RET4:%.*]] = xor i32 [[RES4]], [[CONV4]]
  // OGCG:  store i32 [[RET4]], ptr %{{.*}}, align 4
  si = __sync_xor_and_fetch(&si, uc);

  // CIR: [[VAL5:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u32i
  // CIR: [[RES5:%.*]] = cir.atomic.fetch xor seq_cst fetch_first {{%.*}}, [[VAL5]] : (!cir.ptr<!u32i>, !u32i) -> !u32i
  // CIR: [[RET5:%.*]] = cir.binop(xor, [[RES5]], [[VAL5]]) : !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw xor ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[RET5:%.*]] = xor i32 [[RES5]], [[CONV5]]
  // LLVM:  store i32 [[RET5]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // OGCG:  [[RES5:%.*]] = atomicrmw xor ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // OGCG:  [[RET5:%.*]] = xor i32 [[RES5]], [[CONV5]]
  // OGCG:  store i32 [[RET5]], ptr %{{.*}}, align 4
  ui = __sync_xor_and_fetch(&ui, uc);

  // CIR: [[VAL6:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s64i
  // CIR: [[RES6:%.*]] = cir.atomic.fetch xor seq_cst fetch_first {{%.*}}, [[VAL6]] : (!cir.ptr<!s64i>, !s64i) -> !s64i
  // CIR: [[RET6:%.*]] = cir.binop(xor, [[RES6]], [[VAL6]]) : !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw xor ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[RET6:%.*]] = xor i64 [[RES6]], [[CONV6]]
  // LLVM:  store i64 [[RET6]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // OGCG:  [[RES6:%.*]] = atomicrmw xor ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // OGCG:  [[RET6:%.*]] = xor i64 [[RES6]], [[CONV6]]
  // OGCG:  store i64 [[RET6]], ptr %{{.*}}, align 8
  sll = __sync_xor_and_fetch(&sll, uc);

  // CIR: [[VAL7:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u64i
  // CIR: [[RES7:%.*]] = cir.atomic.fetch xor seq_cst fetch_first {{%.*}}, [[VAL7]] : (!cir.ptr<!u64i>, !u64i) -> !u64i
  // CIR: [[RET7:%.*]] = cir.binop(xor, [[RES7]], [[VAL7]]) : !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw xor ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[RET7:%.*]] = xor i64 [[RES7]], [[CONV7]]
  // LLVM:  store i64 [[RET7]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // OGCG:  [[RES7:%.*]] = atomicrmw xor ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // OGCG:  [[RET7:%.*]] = xor i64 [[RES7]], [[CONV7]]
  // OGCG:  store i64 [[RET7]], ptr %{{.*}}, align 8
  ull = __sync_xor_and_fetch(&ull, uc);

  // CIR: [[VAL0:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s8i
  // CIR: [[RES0:%.*]] = cir.atomic.fetch nand seq_cst fetch_first {{%.*}}, [[VAL0]] : (!cir.ptr<!s8i>, !s8i) -> !s8i
  // CIR: [[INTERM0:%.*]] = cir.binop(and, [[RES0]], [[VAL0]]) : !s8i
  // CIR: [[RET0:%.*]] =  cir.unary(not, [[INTERM0]]) : !s8i, !s8i
  // LLVM:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES0:%.*]] = atomicrmw nand ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // LLVM:  [[INTERM0:%.*]] = and i8 [[RES0]], [[VAL0]]
  // LLVM:  [[RET0:%.*]] = xor i8 [[INTERM0]], -1
  // LLVM:  store i8 [[RET0]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL0:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES0:%.*]] = atomicrmw nand ptr %{{.*}}, i8 [[VAL0]] seq_cst, align 1
  // OGCG:  [[INTERM0:%.*]] = and i8 [[RES0]], [[VAL0]]
  // OGCG:  [[RET0:%.*]] = xor i8 [[INTERM0]], -1
  // OGCG:  store i8 [[RET0]], ptr %{{.*}}, align 1
  sc = __sync_nand_and_fetch(&sc, uc);

  // CIR: [[RES1:%.*]] = cir.atomic.fetch nand seq_cst fetch_first {{%.*}}, [[VAL1:%.*]] : (!cir.ptr<!u8i>, !u8i) -> !u8i
  // CIR: [[INTERM1:%.*]] = cir.binop(and, [[RES1]], [[VAL1]]) : !u8i
  // CIR: [[RET1:%.*]] = cir.unary(not, [[INTERM1]]) : !u8i, !u8i
  // LLVM:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[RES1:%.*]] = atomicrmw nand ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // LLVM:  [[INTERM1:%.*]] = and i8 [[RES1]], [[VAL1]]
  // LLVM:  [[RET1:%.*]] = xor i8 [[INTERM1]], -1
  // LLVM:  store i8 [[RET1]], ptr %{{.*}}, align 1
  // OGCG:  [[VAL1:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[RES1:%.*]] = atomicrmw nand ptr %{{.*}}, i8 [[VAL1]] seq_cst, align 1
  // OGCG:  [[INTERM1:%.*]] = and i8 [[RES1]], [[VAL1]]
  // OGCG:  [[RET1:%.*]] = xor i8 [[INTERM1]], -1
  // OGCG:  store i8 [[RET1]], ptr %{{.*}}, align 1
  uc = __sync_nand_and_fetch(&uc, uc);

  // CIR: [[VAL2:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s16i
  // CIR: [[RES2:%.*]] = cir.atomic.fetch nand seq_cst fetch_first {{%.*}}, [[VAL2]] : (!cir.ptr<!s16i>, !s16i) -> !s16i
  // CIR: [[INTERM2:%.*]] = cir.binop(and, [[RES2]], [[VAL2]]) : !s16i
  // CIR: [[RET2:%.*]] =  cir.unary(not, [[INTERM2]]) : !s16i, !s16i
  // LLVM:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // LLVM:  [[RES2:%.*]] = atomicrmw nand ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // LLVM:  [[INTERM2:%.*]] = and i16 [[RES2]], [[CONV2]]
  // LLVM:  [[RET2:%.*]] = xor i16 [[INTERM2]], -1
  // LLVM:  store i16 [[RET2]], ptr %{{.*}}, align 2
  // OGCG:  [[VAL2:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV2:%.*]] = zext i8 [[VAL2]] to i16
  // OGCG:  [[RES2:%.*]] = atomicrmw nand ptr %{{.*}}, i16 [[CONV2]] seq_cst, align 2
  // OGCG:  [[INTERM2:%.*]] = and i16 [[RES2]], [[CONV2]]
  // OGCG:  [[RET2:%.*]] = xor i16 [[INTERM2]], -1
  // OGCG:  store i16 [[RET2]], ptr %{{.*}}, align 2
  ss = __sync_nand_and_fetch(&ss, uc);

  // CIR: [[VAL3:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u16i
  // CIR: [[RES3:%.*]] = cir.atomic.fetch nand seq_cst fetch_first {{%.*}}, [[VAL3]] : (!cir.ptr<!u16i>, !u16i) -> !u16i
  // CIR: [[INTERM3:%.*]] = cir.binop(and, [[RES3]], [[VAL3]]) : !u16i
  // CIR: [[RET3:%.*]] =  cir.unary(not, [[INTERM3]]) : !u16i, !u16i
  // LLVM:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // LLVM:  [[RES3:%.*]] = atomicrmw nand ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // LLVM:  [[INTERM3:%.*]] = and i16 [[RES3]], [[CONV3]]
  // LLVM:  [[RET3:%.*]] = xor i16 [[INTERM3]], -1
  // LLVM:  store i16 [[RET3]], ptr %{{.*}}, align 2
  // OGCG:  [[VAL3:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV3:%.*]] = zext i8 [[VAL3]] to i16
  // OGCG:  [[RES3:%.*]] = atomicrmw nand ptr %{{.*}}, i16 [[CONV3]] seq_cst, align 2
  // OGCG:  [[INTERM3:%.*]] = and i16 [[RES3]], [[CONV3]]
  // OGCG:  [[RET3:%.*]] = xor i16 [[INTERM3]], -1
  // OGCG:  store i16 [[RET3]], ptr %{{.*}}, align 2
  us = __sync_nand_and_fetch(&us, uc);

  // CIR: [[VAL4:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s32i
  // CIR: [[RES4:%.*]] = cir.atomic.fetch nand seq_cst fetch_first {{%.*}}, [[VAL4]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: [[INTERM4:%.*]] = cir.binop(and, [[RES4]], [[VAL4]]) : !s32i
  // CIR: [[RET4:%.*]] =  cir.unary(not, [[INTERM4]]) : !s32i, !s32i
  // LLVM:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // LLVM:  [[RES4:%.*]] = atomicrmw nand ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // LLVM:  [[INTERM4:%.*]] = and i32 [[RES4]], [[CONV4]]
  // LLVM:  [[RET4:%.*]] = xor i32 [[INTERM4]], -1
  // LLVM:  store i32 [[RET4]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL4:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV4:%.*]] = zext i8 [[VAL4]] to i32
  // OGCG:  [[RES4:%.*]] = atomicrmw nand ptr %{{.*}}, i32 [[CONV4]] seq_cst, align 4
  // OGCG:  [[INTERM4:%.*]] = and i32 [[RES4]], [[CONV4]]
  // OGCG:  [[RET4:%.*]] = xor i32 [[INTERM4]], -1
  // OGCG:  store i32 [[RET4]], ptr %{{.*}}, align 4
  si = __sync_nand_and_fetch(&si, uc);

  // CIR: [[VAL5:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u32i
  // CIR: [[RES5:%.*]] = cir.atomic.fetch nand seq_cst fetch_first {{%.*}}, [[VAL5]] : (!cir.ptr<!u32i>, !u32i) -> !u32i
  // CIR: [[INTERM5:%.*]] = cir.binop(and, [[RES5]], [[VAL5]]) : !u32i
  // CIR: [[RET5:%.*]] =  cir.unary(not, [[INTERM5]]) : !u32i, !u32i
  // LLVM:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // LLVM:  [[RES5:%.*]] = atomicrmw nand ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // LLVM:  [[INTERM5:%.*]] = and i32 [[RES5]], [[CONV5]]
  // LLVM:  [[RET5:%.*]] = xor i32 [[INTERM5]], -1
  // LLVM:  store i32 [[RET5]], ptr %{{.*}}, align 4
  // OGCG:  [[VAL5:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV5:%.*]] = zext i8 [[VAL5]] to i32
  // OGCG:  [[RES5:%.*]] = atomicrmw nand ptr %{{.*}}, i32 [[CONV5]] seq_cst, align 4
  // OGCG:  [[INTERM5:%.*]] = and i32 [[RES5]], [[CONV5]]
  // OGCG:  [[RET5:%.*]] = xor i32 [[INTERM5]], -1
  // OGCG:  store i32 [[RET5]], ptr %{{.*}}, align 4
  ui = __sync_nand_and_fetch(&ui, uc);

  // CIR: [[VAL6:%.*]] = cir.cast integral {{%.*}} : !u8i -> !s64i
  // CIR: [[RES6:%.*]] = cir.atomic.fetch nand seq_cst fetch_first {{%.*}}, [[VAL6]] : (!cir.ptr<!s64i>, !s64i) -> !s64i
  // CIR: [[INTERM6:%.*]] = cir.binop(and, [[RES6]], [[VAL6]]) : !s64i
  // CIR: [[RET6:%.*]] =  cir.unary(not, [[INTERM6]]) : !s64i, !s64i
  // LLVM:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // LLVM:  [[RES6:%.*]] = atomicrmw nand ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // LLVM:  [[INTERM6:%.*]] = and i64 [[RES6]], [[CONV6]]
  // LLVM:  [[RET6:%.*]] = xor i64 [[INTERM6]], -1
  // LLVM:  store i64 [[RET6]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL6:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV6:%.*]] = zext i8 [[VAL6]] to i64
  // OGCG:  [[RES6:%.*]] = atomicrmw nand ptr %{{.*}}, i64 [[CONV6]] seq_cst, align 8
  // OGCG:  [[INTERM6:%.*]] = and i64 [[RES6]], [[CONV6]]
  // OGCG:  [[RET6:%.*]] = xor i64 [[INTERM6]], -1
  // OGCG:  store i64 [[RET6]], ptr %{{.*}}, align 8
  sll = __sync_nand_and_fetch(&sll, uc);

  // CIR: [[VAL7:%.*]] = cir.cast integral {{%.*}} : !u8i -> !u64i
  // CIR: [[RES7:%.*]] = cir.atomic.fetch nand seq_cst fetch_first {{%.*}}, [[VAL7]] : (!cir.ptr<!u64i>, !u64i) -> !u64i
  // CIR: [[INTERM7:%.*]] = cir.binop(and, [[RES7]], [[VAL7]]) : !u64i
  // CIR: [[RET7:%.*]] =  cir.unary(not, [[INTERM7]]) : !u64i, !u64i
  // LLVM:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // LLVM:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // LLVM:  [[RES7:%.*]] = atomicrmw nand ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // LLVM:  [[INTERM7:%.*]] = and i64 [[RES7]], [[CONV7]]
  // LLVM:  [[RET7:%.*]] = xor i64 [[INTERM7]], -1
  // LLVM:  store i64 [[RET7]], ptr %{{.*}}, align 8
  // OGCG:  [[VAL7:%.*]] = load i8, ptr %{{.*}}, align 1
  // OGCG:  [[CONV7:%.*]] = zext i8 [[VAL7]] to i64
  // OGCG:  [[RES7:%.*]] = atomicrmw nand ptr %{{.*}}, i64 [[CONV7]] seq_cst, align 8
  // OGCG:  [[INTERM7:%.*]] = and i64 [[RES7]], [[CONV7]]
  // OGCG:  [[RET7:%.*]] = xor i64 [[INTERM7]], -1
  // OGCG:  store i64 [[RET7]], ptr %{{.*}}, align 8
  ull = __sync_nand_and_fetch(&ull, uc);
}

int atomic_load_dynamic_order(int *ptr, int order) {
  // CIR-LABEL: atomic_load_dynamic_order
  // LLVM-LABEL: atomic_load_dynamic_order
  // OGCG-LABEL: atomic_load_dynamic_order

  return __atomic_load_n(ptr, order);
  
  // CIR:      %[[PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR-NEXT: %[[ORDER:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT: cir.switch(%[[ORDER]] : !s32i) {
  // CIR-NEXT:   cir.case(default, []) {
  // CIR-NEXT:     %[[RES:.+]] = cir.load align(4) syncscope(system) atomic(relaxed) %[[PTR]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     cir.store align(4) %[[RES]], %[[RES_SLOT:.+]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
  // CIR-NEXT:     %[[RES:.+]] = cir.load align(4) syncscope(system) atomic(acquire) %[[PTR]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     cir.store align(4) %[[RES]], %[[RES_SLOT]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.case(anyof, [#cir.int<5> : !s32i]) {
  // CIR-NEXT:     %[[RES:.+]] = cir.load align(4) syncscope(system) atomic(seq_cst) %[[PTR]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     cir.store align(4) %[[RES]], %[[RES_SLOT]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.yield
  // CIR-NEXT: }
  // CIR-NEXT: %{{.+}} = cir.load align(4) %[[RES_SLOT]] : !cir.ptr<!s32i>, !s32i

  // LLVM:        %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM-NEXT:   %[[ORDER:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   br label %[[SWITCH_BLK:.+]]
  // LLVM:      [[SWITCH_BLK]]:
  // LLVM-NEXT:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // LLVM-NEXT:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // LLVM-NEXT:     i32 2, label %[[ACQUIRE_BLK]]
  // LLVM-NEXT:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // LLVM-NEXT:   ]
  // LLVM:      [[DEFAULT_BLK]]:
  // LLVM-NEXT:   %[[RES:.+]] = load atomic i32, ptr %[[PTR]] monotonic, align 4
  // LLVM-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT:.+]], align 4
  // LLVM-NEXT:   br label %[[CONTINUE_BLK:.+]]
  // LLVM:      [[ACQUIRE_BLK]]:
  // LLVM-NEXT:   %[[RES:.+]] = load atomic i32, ptr %[[PTR]] acquire, align 4
  // LLVM-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // LLVM-NEXT:   br label %[[CONTINUE_BLK]]
  // LLVM:      [[SEQ_CST_BLK]]:
  // LLVM-NEXT:   %[[RES:.+]] = load atomic i32, ptr %[[PTR]] seq_cst, align 4
  // LLVM-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // LLVM-NEXT:   br label %[[CONTINUE_BLK]]
  // LLVM:      [[CONTINUE_BLK]]:
  // LLVM-NEXT:   %{{.+}} = load i32, ptr %[[RES_SLOT]], align 4

  // OGCG:        %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // OGCG-NEXT:   %[[ORDER:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // OGCG-NEXT:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // OGCG-NEXT:     i32 2, label %[[ACQUIRE_BLK]]
  // OGCG-NEXT:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // OGCG-NEXT:   ]
  // OGCG:      [[DEFAULT_BLK]]:
  // OGCG-NEXT:   %[[RES:.+]] = load atomic i32, ptr %[[PTR]] monotonic, align 4
  // OGCG-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT:.+]], align 4
  // OGCG-NEXT:   br label %[[CONTINUE_BLK:.+]]
  // OGCG:      [[ACQUIRE_BLK]]:
  // OGCG-NEXT:   %[[RES:.+]] = load atomic i32, ptr %[[PTR]] acquire, align 4
  // OGCG-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // OGCG-NEXT:   br label %[[CONTINUE_BLK]]
  // OGCG:      [[SEQ_CST_BLK]]:
  // OGCG-NEXT:   %[[RES:.+]] = load atomic i32, ptr %[[PTR]] seq_cst, align 4
  // OGCG-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // OGCG-NEXT:   br label %[[CONTINUE_BLK]]
  // OGCG:      [[CONTINUE_BLK]]:
  // OGCG-NEXT:   %{{.+}} = load i32, ptr %[[RES_SLOT]], align 4
}

void atomic_store_dynamic_order(int *ptr, int order) {
  // CIR-LABEL: atomic_store_dynamic_order
  // LLVM-LABEL: atomic_store_dynamic_order
  // OGCG-LABEL: atomic_store_dynamic_order

  __atomic_store_n(ptr, 10, order);

  // CIR:      %[[PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR:      %[[ORDER:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR:      cir.switch(%[[ORDER]] : !s32i) {
  // CIR-NEXT:   cir.case(default, []) {
  // CIR-NEXT:     %[[VALUE:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     cir.store align(4) syncscope(system) atomic(relaxed) %[[VALUE]], %[[PTR]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.case(anyof, [#cir.int<3> : !s32i]) {
  // CIR-NEXT:     %[[VALUE:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     cir.store align(4) syncscope(system) atomic(release) %[[VALUE]], %[[PTR]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.case(anyof, [#cir.int<5> : !s32i]) {
  // CIR-NEXT:     %[[VALUE:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     cir.store align(4) syncscope(system) atomic(seq_cst) %[[VALUE]], %[[PTR]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.yield
  // CIR-NEXT: }

  // LLVM:        %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM:        %[[ORDER:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM:        br label %[[SWITCH_BLK:.+]]
  // LLVM:      [[SWITCH_BLK]]:
  // LLVM-NEXT:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // LLVM-NEXT:     i32 3, label %[[RELEASE_BLK:.+]]
  // LLVM-NEXT:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // LLVM-NEXT:   ]
  // LLVM:      [[DEFAULT_BLK]]:
  // LLVM-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   store atomic i32 %[[VALUE]], ptr %[[PTR]] monotonic, align 4
  // LLVM-NEXT:   br label %{{.+}}
  // LLVM:      [[RELEASE_BLK]]:
  // LLVM-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   store atomic i32 %[[VALUE]], ptr %[[PTR]] release, align 4
  // LLVM-NEXT:   br label %{{.+}}
  // LLVM:      [[SEQ_CST_BLK]]:
  // LLVM-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   store atomic i32 %[[VALUE]], ptr %[[PTR]] seq_cst, align 4
  // LLVM-NEXT:   br label %{{.+}}
  
  // OGCG:        %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // OGCG-NEXT:   %[[ORDER:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG:        switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // OGCG-NEXT:     i32 3, label %[[RELEASE_BLK:.+]]
  // OGCG-NEXT:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // OGCG-NEXT:   ]
  // OGCG:      [[DEFAULT_BLK]]:
  // OGCG-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   store atomic i32 %[[VALUE]], ptr %[[PTR]] monotonic, align 4
  // OGCG-NEXT:   br label %{{.+}}
  // OGCG:      [[RELEASE_BLK]]:
  // OGCG-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   store atomic i32 %[[VALUE]], ptr %[[PTR]] release, align 4
  // OGCG-NEXT:   br label %{{.+}}
  // OGCG:      [[SEQ_CST_BLK]]:
  // OGCG-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   store atomic i32 %[[VALUE]], ptr %[[PTR]] seq_cst, align 4
  // OGCG-NEXT:   br label %{{.+}}
}

int atomic_load_and_store_dynamic_order(int *ptr, int order) {
  // CIR-LABEL: atomic_load_and_store_dynamic_order
  // LLVM-LABEL: atomic_load_and_store_dynamic_order
  // OGCG-LABEL: atomic_load_and_store_dynamic_order

  return __atomic_exchange_n(ptr, 20, order);

  // CIR:      %[[PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR:      %[[ORDER:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR:      cir.switch(%[[ORDER]] : !s32i) {
  // CIR-NEXT:   cir.case(default, []) {
  // CIR-NEXT:     %[[LIT:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     %[[RES:.+]] = cir.atomic.xchg relaxed %[[PTR]], %[[LIT]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR-NEXT:     cir.store align(4) %[[RES]], %[[RES_SLOT:.+]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
  // CIR-NEXT:     %[[LIT:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     %[[RES:.+]] = cir.atomic.xchg acquire %[[PTR]], %[[LIT]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR-NEXT:     cir.store align(4) %[[RES]], %[[RES_SLOT]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.case(anyof, [#cir.int<3> : !s32i]) {
  // CIR-NEXT:     %[[LIT:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     %[[RES:.+]] = cir.atomic.xchg release %[[PTR]], %[[LIT]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR-NEXT:     cir.store align(4) %[[RES]], %[[RES_SLOT]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.case(anyof, [#cir.int<4> : !s32i]) {
  // CIR-NEXT:     %[[LIT:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     %[[RES:.+]] = cir.atomic.xchg acq_rel %[[PTR]], %[[LIT]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR-NEXT:     cir.store align(4) %[[RES]], %[[RES_SLOT]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.case(anyof, [#cir.int<5> : !s32i]) {
  // CIR-NEXT:     %[[LIT:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:     %[[RES:.+]] = cir.atomic.xchg seq_cst %[[PTR]], %[[LIT]] : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR-NEXT:     cir.store align(4) %[[RES]], %[[RES_SLOT]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:     cir.break
  // CIR-NEXT:   }
  // CIR-NEXT:   cir.yield
  // CIR-NEXT: }
  // CIR-NEXT: %{{.+}} = cir.load align(4) %[[RES_SLOT]] : !cir.ptr<!s32i>, !s32i

  // LLVM:        %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // LLVM:        %[[ORDER:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM:        br label %[[SWITCH_BLK:.+]]
  // LLVM:      [[SWITCH_BLK]]:
  // LLVM-NEXT:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // LLVM-NEXT:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // LLVM-NEXT:     i32 2, label %[[ACQUIRE_BLK]]
  // LLVM-NEXT:     i32 3, label %[[RELEASE_BLK:.+]]
  // LLVM-NEXT:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // LLVM-NEXT:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // LLVM-NEXT:   ]
  // LLVM:      [[DEFAULT_BLK]]:
  // LLVM-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] monotonic, align 4
  // LLVM-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT:.+]], align 4
  // LLVM-NEXT:   br label %[[CONTINUE_BLK:.+]]
  // LLVM:      [[ACQUIRE_BLK]]:
  // LLVM-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] acquire, align 4
  // LLVM-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // LLVM-NEXT:   br label %[[CONTINUE_BLK]]
  // LLVM:      [[RELEASE_BLK]]:
  // LLVM-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] release, align 4
  // LLVM-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // LLVM-NEXT:   br label %[[CONTINUE_BLK]]
  // LLVM:      [[ACQ_REL_BLK]]:
  // LLVM-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] acq_rel, align 4
  // LLVM-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // LLVM-NEXT:   br label %[[CONTINUE_BLK]]
  // LLVM:      [[SEQ_CST_BLK]]:
  // LLVM-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // LLVM-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] seq_cst, align 4
  // LLVM-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // LLVM-NEXT:   br label %[[CONTINUE_BLK]]
  // LLVM:      [[CONTINUE_BLK]]:
  // LLVM-NEXT:   %{{.+}} = load i32, ptr %[[RES_SLOT]], align 4
  
  // OGCG:        %[[PTR:.+]] = load ptr, ptr %{{.+}}, align 8
  // OGCG-NEXT:   %[[ORDER:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG:        switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // OGCG-NEXT:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // OGCG-NEXT:     i32 2, label %[[ACQUIRE_BLK]]
  // OGCG-NEXT:     i32 3, label %[[RELEASE_BLK:.+]]
  // OGCG-NEXT:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // OGCG-NEXT:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // OGCG-NEXT:   ]
  // OGCG:      [[DEFAULT_BLK]]:
  // OGCG-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] monotonic, align 4
  // OGCG-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT:.+]], align 4
  // OGCG-NEXT:   br label %[[CONTINUE_BLK:.+]]
  // OGCG:      [[ACQUIRE_BLK]]:
  // OGCG-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] acquire, align 4
  // OGCG-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // OGCG-NEXT:   br label %[[CONTINUE_BLK]]
  // OGCG:      [[RELEASE_BLK]]:
  // OGCG-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] release, align 4
  // OGCG-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // OGCG-NEXT:   br label %[[CONTINUE_BLK]]
  // OGCG:      [[ACQ_REL_BLK]]:
  // OGCG-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] acq_rel, align 4
  // OGCG-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // OGCG-NEXT:   br label %[[CONTINUE_BLK]]
  // OGCG:      [[SEQ_CST_BLK]]:
  // OGCG-NEXT:   %[[VALUE:.+]] = load i32, ptr %{{.+}}, align 4
  // OGCG-NEXT:   %[[RES:.+]] = atomicrmw xchg ptr %[[PTR]], i32 %[[VALUE]] seq_cst, align 4
  // OGCG-NEXT:   store i32 %[[RES]], ptr %[[RES_SLOT]], align 4
  // OGCG-NEXT:   br label %[[CONTINUE_BLK]]
  // OGCG:      [[CONTINUE_BLK]]:
  // OGCG-NEXT:   %{{.+}} = load i32, ptr %[[RES_SLOT]], align 4
}
