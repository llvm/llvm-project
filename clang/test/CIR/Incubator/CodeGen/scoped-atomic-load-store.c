// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int scoped_load_thread(int *ptr) {
  return __scoped_atomic_load_n(ptr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
}

// CIR-LABEL: @scoped_load_thread
// CIR: %[[ATOMIC_LOAD:.*]] = cir.load align(4) syncscope(single_thread) atomic(relaxed) %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.store align(4) %[[ATOMIC_LOAD]], %{{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: @scoped_load_thread
// LLVM: load atomic i32, ptr %{{.*}} syncscope("singlethread") monotonic, align 4

// OGCG-LABEL: @scoped_load_thread
// OGCG: load atomic i32, ptr %{{.*}} monotonic, align 4

int scoped_load_system(int *ptr) {
  return __scoped_atomic_load_n(ptr, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM);
}

// CIR-LABEL: @scoped_load_system
// CIR: cir.load align(4) syncscope(system) atomic(seq_cst) %{{.*}} : !cir.ptr<!s32i>, !s32i

// LLVM-LABEL: @scoped_load_system
// LLVM: load atomic i32, ptr %{{.*}} seq_cst, align 4
// LLVM-NOT: syncscope(

// OGCG-LABEL: @scoped_load_system
// OGCG: load atomic i32, ptr %{{.*}} seq_cst, align 4
// OGCG-NOT: syncscope(
