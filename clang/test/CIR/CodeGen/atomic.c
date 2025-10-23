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

