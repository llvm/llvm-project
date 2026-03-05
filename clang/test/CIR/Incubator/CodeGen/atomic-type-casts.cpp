// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// Test CK_AtomicToNonAtomic and CK_NonAtomicToAtomic casts
// Note: Full atomic load/store support is NYI - this tests just the casts

// Test NonAtomicToAtomic cast (assigning non-atomic to atomic)
void test_non_atomic_to_atomic() {
  int x = 50;
  _Atomic int y = x;  // Implicit NonAtomicToAtomic cast
  // CIR: cir.func{{.*}}test_non_atomic_to_atomicv
  // CIR: cir.alloca !s32i, !cir.ptr<!s32i>, ["x"
  // CIR: cir.alloca !s32i, !cir.ptr<!s32i>, ["y"
  // CIR: cir.load
  // CIR: cir.store
  // LLVM-LABEL: @_Z25test_non_atomic_to_atomicv
  // LLVM: alloca i32
  // LLVM: alloca i32
  // LLVM: store i32 50
  // LLVM: load i32
  // LLVM: store i32
  // OGCG-LABEL: @_Z25test_non_atomic_to_atomicv
  // OGCG: %x = alloca i32
  // OGCG: %y = alloca i32
  // OGCG: store i32 50
}

// Test that atomic type casts don't crash the compiler
void test_atomic_cast_exists() {
  int regular = 42;
  _Atomic int atomic_val = regular;
  // Just verify this compiles - the cast infrastructure exists
  // CIR: cir.func{{.*}}test_atomic_cast_existsv
  // CIR: cir.alloca !s32i, !cir.ptr<!s32i>, ["regular"
  // CIR: cir.alloca !s32i, !cir.ptr<!s32i>, ["atomic_val"
  // LLVM-LABEL: @_Z23test_atomic_cast_existsv
  // LLVM: alloca i32
  // LLVM: alloca i32
  // LLVM: store i32 42
  // OGCG-LABEL: @_Z23test_atomic_cast_existsv
  // OGCG: %regular = alloca i32
  // OGCG: %atomic_val = alloca i32
  // OGCG: store i32 42
}

// Test with different types
void test_atomic_float_cast() {
  float f = 3.14f;
  _Atomic float g = f;
  // CIR: cir.func{{.*}}test_atomic_float_castv
  // CIR: cir.alloca !cir.float
  // CIR: cir.alloca !cir.float
  // LLVM-LABEL: @_Z22test_atomic_float_castv
  // LLVM: alloca float
  // LLVM: alloca float
  // LLVM: store float
  // OGCG-LABEL: @_Z22test_atomic_float_castv
  // OGCG: %f = alloca float
  // OGCG: %g = alloca float
  // OGCG: store float
}

// Test that cast infrastructure is in place for pointers
void test_atomic_pointer_cast() {
  int val = 42;
  int* ptr = &val;
  _Atomic(int*) atomic_ptr = ptr;
  // CIR: cir.func{{.*}}test_atomic_pointer_castv
  // CIR: cir.alloca !cir.ptr<!s32i>
  // CIR: cir.alloca !cir.ptr<!s32i>
  // LLVM-LABEL: @_Z24test_atomic_pointer_castv
  // LLVM: alloca i32
  // LLVM: alloca ptr
  // LLVM: alloca ptr
  // LLVM: store i32 42
  // OGCG-LABEL: @_Z24test_atomic_pointer_castv
  // OGCG: %val = alloca i32
  // OGCG: %ptr = alloca ptr
  // OGCG: %atomic_ptr = alloca ptr
  // OGCG: store i32 42
}
