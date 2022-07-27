// RUN: %clang_cc1 -no-opaque-pointers -Werror -Wno-atomic-alignment -triple powerpc64le-linux-gnu \
// RUN:   -target-cpu pwr8 -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC64-QUADWORD-ATOMICS
// RUN: %clang_cc1 -no-opaque-pointers -Werror -Wno-atomic-alignment -triple powerpc64le-linux-gnu \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC64
// RUN: %clang_cc1 -no-opaque-pointers -Werror -Wno-atomic-alignment -triple powerpc64-unknown-aix \
// RUN:   -target-cpu pwr7 -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC64
// RUN: %clang_cc1 -no-opaque-pointers -Werror -Wno-atomic-alignment -triple powerpc64-unknown-aix \
// RUN:   -target-cpu pwr8 -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC64
// RUN: %clang_cc1 -no-opaque-pointers -Werror -Wno-atomic-alignment -triple powerpc64-unknown-aix \
// RUN:   -mabi=quadword-atomics -target-cpu pwr8 -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefix=PPC64-QUADWORD-ATOMICS


typedef struct {
  char x[16];
} Q;

typedef _Atomic(Q) AtomicQ;

typedef __int128_t int128_t;

// PPC64-QUADWORD-ATOMICS-LABEL: @test_load(
// PPC64-QUADWORD-ATOMICS:    [[TMP3:%.*]] = load atomic i128, i128* [[TMP1:%.*]] acquire, align 16
//
// PPC64-LABEL: @test_load(
// PPC64:    call void @__atomic_load(i64 noundef 16, i8* noundef [[TMP3:%.*]], i8* noundef [[TMP4:%.*]], i32 noundef signext 2)
//
Q test_load(AtomicQ *ptr) {
  // expected-no-diagnostics
  return __c11_atomic_load(ptr, __ATOMIC_ACQUIRE);
}

// PPC64-QUADWORD-ATOMICS-LABEL: @test_store(
// PPC64-QUADWORD-ATOMICS:    store atomic i128 [[TMP6:%.*]], i128* [[TMP4:%.*]] release, align 16
//
// PPC64-LABEL: @test_store(
// PPC64:    call void @__atomic_store(i64 noundef 16, i8* noundef [[TMP6:%.*]], i8* noundef [[TMP7:%.*]], i32 noundef signext 3)
//
void test_store(Q val, AtomicQ *ptr) {
  // expected-no-diagnostics
  __c11_atomic_store(ptr, val, __ATOMIC_RELEASE);
}

// PPC64-QUADWORD-ATOMICS-LABEL: @test_add(
// PPC64-QUADWORD-ATOMICS:    [[TMP3:%.*]] = atomicrmw add i128* [[TMP0:%.*]], i128 [[TMP2:%.*]] monotonic, align 16
//
// PPC64-LABEL: @test_add(
// PPC64:    [[CALL:%.*]] = call i128 @__atomic_fetch_add_16(i8* noundef [[TMP2:%.*]], i128 noundef [[TMP3:%.*]], i32 noundef signext 0)
//
void test_add(_Atomic(int128_t) *ptr, int128_t x) {
  // expected-no-diagnostics
  __c11_atomic_fetch_add(ptr, x, __ATOMIC_RELAXED);
}

// PPC64-QUADWORD-ATOMICS-LABEL: @test_xchg(
// PPC64-QUADWORD-ATOMICS:    [[TMP8:%.*]] = atomicrmw xchg i128* [[TMP4:%.*]], i128 [[TMP7:%.*]] seq_cst, align 16
//
// PPC64-LABEL: @test_xchg(
// PPC64:    call void @__atomic_exchange(i64 noundef 16, i8* noundef [[TMP7:%.*]], i8* noundef [[TMP8:%.*]], i8* noundef [[TMP9:%.*]], i32 noundef signext 5)
//
Q test_xchg(AtomicQ *ptr, Q new) {
  // expected-no-diagnostics
  return __c11_atomic_exchange(ptr, new, __ATOMIC_SEQ_CST);
}

// PPC64-QUADWORD-ATOMICS-LABEL: @test_cmpxchg(
// PPC64-QUADWORD-ATOMICS:    [[TMP10:%.*]] = cmpxchg i128* [[TMP5:%.*]], i128 [[TMP8:%.*]], i128 [[TMP9:%.*]] seq_cst monotonic, align 16
//
// PPC64-LABEL: @test_cmpxchg(
// PPC64:    [[CALL:%.*]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 16, i8* noundef [[TMP8:%.*]], i8* noundef [[TMP9:%.*]], i8* noundef [[TMP10:%.*]], i32 noundef signext 5, i32 noundef signext 0)
//
int test_cmpxchg(AtomicQ *ptr, Q *cmp, Q new) {
  // expected-no-diagnostics
  return __c11_atomic_compare_exchange_strong(ptr, cmp, new, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
}

// PPC64-QUADWORD-ATOMICS-LABEL: @test_cmpxchg_weak(
// PPC64-QUADWORD-ATOMICS:    [[TMP10:%.*]] = cmpxchg weak i128* [[TMP5:%.*]], i128 [[TMP8:%.*]], i128 [[TMP9:%.*]] seq_cst monotonic, align 16
//
// PPC64-LABEL: @test_cmpxchg_weak(
// PPC64:    [[CALL:%.*]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 16, i8* noundef [[TMP8:%.*]], i8* noundef [[TMP9:%.*]], i8* noundef [[TMP10:%.*]], i32 noundef signext 5, i32 noundef signext 0)
//
int test_cmpxchg_weak(AtomicQ *ptr, Q *cmp, Q new) {
  // expected-no-diagnostics
  return __c11_atomic_compare_exchange_weak(ptr, cmp, new, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
}

// PPC64-QUADWORD-ATOMICS-LABEL: @is_lock_free(
// PPC64-QUADWORD-ATOMICS:    ret i32 1
//
// PPC64-LABEL: @is_lock_free(
// PPC64:    [[CALL:%.*]] = call zeroext i1 @__atomic_is_lock_free(i64 noundef 16, i8* noundef null)
//
int is_lock_free() {
  AtomicQ q;
 // expected-no-diagnostics
  return __c11_atomic_is_lock_free(sizeof(q));
}
