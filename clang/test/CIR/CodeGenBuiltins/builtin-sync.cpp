// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

extern "C" {

// __sync_val_compare_and_swap

// CIR-LABEL: @test_sync_val_compare_and_swap_1(
// CIR: cir.atomic.cmpxchg success(seq_cst) failure(seq_cst) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} : (!cir.ptr<!s8i>, !s8i, !s8i) -> (!s8i, !cir.bool)
// LLVM-LABEL: @test_sync_val_compare_and_swap_1(
// LLVM: cmpxchg ptr %{{.+}}, i8 %{{.+}}, i8 %{{.+}} seq_cst seq_cst, align 1
char test_sync_val_compare_and_swap_1(char *p, char oldv, char newv) {
  return __sync_val_compare_and_swap(p, oldv, newv);
}

// CIR-LABEL: @test_sync_val_compare_and_swap_2(
// CIR: cir.atomic.cmpxchg success(seq_cst) failure(seq_cst) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} : (!cir.ptr<!s16i>, !s16i, !s16i) -> (!s16i, !cir.bool)
// LLVM-LABEL: @test_sync_val_compare_and_swap_2(
// LLVM: cmpxchg ptr %{{.+}}, i16 %{{.+}}, i16 %{{.+}} seq_cst seq_cst, align 2
short test_sync_val_compare_and_swap_2(short *p, short oldv, short newv) {
  return __sync_val_compare_and_swap(p, oldv, newv);
}

// CIR-LABEL: @test_sync_val_compare_and_swap_4(
// CIR: cir.atomic.cmpxchg success(seq_cst) failure(seq_cst) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
// LLVM-LABEL: @test_sync_val_compare_and_swap_4(
// LLVM: cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst seq_cst, align 4
int test_sync_val_compare_and_swap_4(int *p, int oldv, int newv) {
  return __sync_val_compare_and_swap(p, oldv, newv);
}

// CIR-LABEL: @test_sync_val_compare_and_swap_8(
// CIR: cir.atomic.cmpxchg success(seq_cst) failure(seq_cst) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} : (!cir.ptr<!s64i>, !s64i, !s64i) -> (!s64i, !cir.bool)
// LLVM-LABEL: @test_sync_val_compare_and_swap_8(
// LLVM: cmpxchg ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} seq_cst seq_cst, align 8
long long test_sync_val_compare_and_swap_8(long long *p, long long oldv,
                                           long long newv) {
  return __sync_val_compare_and_swap(p, oldv, newv);
}

// __sync_bool_compare_and_swap

// CIR-LABEL: @test_sync_bool_compare_and_swap_1(
// CIR: cir.atomic.cmpxchg success(seq_cst) failure(seq_cst) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} : (!cir.ptr<!s8i>, !s8i, !s8i) -> (!s8i, !cir.bool)
// LLVM-LABEL: @test_sync_bool_compare_and_swap_1(
// LLVM: cmpxchg ptr %{{.+}}, i8 %{{.+}}, i8 %{{.+}} seq_cst seq_cst, align 1
// LLVM: extractvalue { i8, i1 } %{{.+}}, 1
bool test_sync_bool_compare_and_swap_1(char *p, char oldv, char newv) {
  return __sync_bool_compare_and_swap(p, oldv, newv);
}

// CIR-LABEL: @test_sync_bool_compare_and_swap_2(
// CIR: cir.atomic.cmpxchg success(seq_cst) failure(seq_cst) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} : (!cir.ptr<!s16i>, !s16i, !s16i) -> (!s16i, !cir.bool)
// LLVM-LABEL: @test_sync_bool_compare_and_swap_2(
// LLVM: cmpxchg ptr %{{.+}}, i16 %{{.+}}, i16 %{{.+}} seq_cst seq_cst, align 2
// LLVM: extractvalue { i16, i1 } %{{.+}}, 1
bool test_sync_bool_compare_and_swap_2(short *p, short oldv, short newv) {
  return __sync_bool_compare_and_swap(p, oldv, newv);
}

// CIR-LABEL: @test_sync_bool_compare_and_swap_4(
// CIR: cir.atomic.cmpxchg success(seq_cst) failure(seq_cst) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool)
// LLVM-LABEL: @test_sync_bool_compare_and_swap_4(
// LLVM: cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} seq_cst seq_cst, align 4
// LLVM: extractvalue { i32, i1 } %{{.+}}, 1
bool test_sync_bool_compare_and_swap_4(int *p, int oldv, int newv) {
  return __sync_bool_compare_and_swap(p, oldv, newv);
}

// CIR-LABEL: @test_sync_bool_compare_and_swap_8(
// CIR: cir.atomic.cmpxchg success(seq_cst) failure(seq_cst) syncscope(system) %{{.+}}, %{{.+}}, %{{.+}} : (!cir.ptr<!s64i>, !s64i, !s64i) -> (!s64i, !cir.bool)
// LLVM-LABEL: @test_sync_bool_compare_and_swap_8(
// LLVM: cmpxchg ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} seq_cst seq_cst, align 8
// LLVM: extractvalue { i64, i1 } %{{.+}}, 1
bool test_sync_bool_compare_and_swap_8(long long *p, long long oldv,
                                       long long newv) {
  return __sync_bool_compare_and_swap(p, oldv, newv);
}

// __sync_swap

// CIR-LABEL: @test_sync_swap_1(
// CIR: cir.atomic.xchg seq_cst syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s8i>, !s8i) -> !s8i
// LLVM-LABEL: @test_sync_swap_1(
// LLVM: atomicrmw xchg ptr %{{.+}}, i8 %{{.+}} seq_cst, align 1
char test_sync_swap_1(char *p, char val) {
  return __sync_swap(p, val);
}

// CIR-LABEL: @test_sync_swap_2(
// CIR: cir.atomic.xchg seq_cst syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s16i>, !s16i) -> !s16i
// LLVM-LABEL: @test_sync_swap_2(
// LLVM: atomicrmw xchg ptr %{{.+}}, i16 %{{.+}} seq_cst, align 2
short test_sync_swap_2(short *p, short val) {
  return __sync_swap(p, val);
}

// CIR-LABEL: @test_sync_swap_4(
// CIR: cir.atomic.xchg seq_cst syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @test_sync_swap_4(
// LLVM: atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
int test_sync_swap_4(int *p, int val) {
  return __sync_swap(p, val);
}

// CIR-LABEL: @test_sync_swap_8(
// CIR: cir.atomic.xchg seq_cst syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @test_sync_swap_8(
// LLVM: atomicrmw xchg ptr %{{.+}}, i64 %{{.+}} seq_cst, align 8
long long test_sync_swap_8(long long *p, long long val) {
  return __sync_swap(p, val);
}

// __sync_lock_test_and_set

// CIR-LABEL: @test_sync_lock_test_and_set_1(
// CIR: cir.atomic.xchg seq_cst syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s8i>, !s8i) -> !s8i
// LLVM-LABEL: @test_sync_lock_test_and_set_1(
// LLVM: atomicrmw xchg ptr %{{.+}}, i8 %{{.+}} seq_cst, align 1
char test_sync_lock_test_and_set_1(char *p, char val) {
  return __sync_lock_test_and_set(p, val);
}

// CIR-LABEL: @test_sync_lock_test_and_set_2(
// CIR: cir.atomic.xchg seq_cst syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s16i>, !s16i) -> !s16i
// LLVM-LABEL: @test_sync_lock_test_and_set_2(
// LLVM: atomicrmw xchg ptr %{{.+}}, i16 %{{.+}} seq_cst, align 2
short test_sync_lock_test_and_set_2(short *p, short val) {
  return __sync_lock_test_and_set(p, val);
}

// CIR-LABEL: @test_sync_lock_test_and_set_4(
// CIR: cir.atomic.xchg seq_cst syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @test_sync_lock_test_and_set_4(
// LLVM: atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} seq_cst, align 4
int test_sync_lock_test_and_set_4(int *p, int val) {
  return __sync_lock_test_and_set(p, val);
}

// CIR-LABEL: @test_sync_lock_test_and_set_8(
// CIR: cir.atomic.xchg seq_cst syncscope(system) %{{.+}}, %{{.+}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @test_sync_lock_test_and_set_8(
// LLVM: atomicrmw xchg ptr %{{.+}}, i64 %{{.+}} seq_cst, align 8
long long test_sync_lock_test_and_set_8(long long *p, long long val) {
  return __sync_lock_test_and_set(p, val);
}

// __sync_lock_release

// CIR-LABEL: @test_sync_lock_release_1(
// CIR: cir.store {{.*}}atomic(release) {{.*}} : !s8i, !cir.ptr<!s8i>
// LLVM-LABEL: @test_sync_lock_release_1(
// LLVM: store atomic i8 0, ptr %{{.+}} release, align 1
void test_sync_lock_release_1(char *p) {
  __sync_lock_release(p);
}

// CIR-LABEL: @test_sync_lock_release_2(
// CIR: cir.store {{.*}}atomic(release) {{.*}} : !s16i, !cir.ptr<!s16i>
// LLVM-LABEL: @test_sync_lock_release_2(
// LLVM: store atomic i16 0, ptr %{{.+}} release, align 2
void test_sync_lock_release_2(short *p) {
  __sync_lock_release(p);
}

// CIR-LABEL: @test_sync_lock_release_4(
// CIR: cir.store {{.*}}atomic(release) {{.*}} : !s32i, !cir.ptr<!s32i>
// LLVM-LABEL: @test_sync_lock_release_4(
// LLVM: store atomic i32 0, ptr %{{.+}} release, align 4
void test_sync_lock_release_4(int *p) {
  __sync_lock_release(p);
}

// CIR-LABEL: @test_sync_lock_release_8(
// CIR: cir.store {{.*}}atomic(release) {{.*}} : !s64i, !cir.ptr<!s64i>
// LLVM-LABEL: @test_sync_lock_release_8(
// LLVM: store atomic i64 0, ptr %{{.+}} release, align 8
void test_sync_lock_release_8(long long *p) {
  __sync_lock_release(p);
}

}
