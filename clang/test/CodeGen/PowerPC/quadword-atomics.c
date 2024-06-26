// RUN: %clang_cc1 -Werror -Wno-atomic-alignment -triple powerpc64le-linux-gnu \
// RUN:   -target-cpu pwr8 -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefixes=PPC64,PPC64-QUADWORD-ATOMICS
// RUN: %clang_cc1 -Werror -Wno-atomic-alignment -triple powerpc64le-linux-gnu \
// RUN:   -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefixes=PPC64,PPC64-NO-QUADWORD-ATOMICS
// RUN: %clang_cc1 -Werror -Wno-atomic-alignment -triple powerpc64-unknown-aix \
// RUN:   -target-cpu pwr7 -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefixes=PPC64,PPC64-NO-QUADWORD-ATOMICS
// RUN: %clang_cc1 -Werror -Wno-atomic-alignment -triple powerpc64-unknown-aix \
// RUN:   -target-cpu pwr8 -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefixes=PPC64,PPC64-NO-QUADWORD-ATOMICS
// RUN: %clang_cc1 -Werror -Wno-atomic-alignment -triple powerpc64-unknown-aix \
// RUN:   -mabi=quadword-atomics -target-cpu pwr8 -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefixes=PPC64,PPC64-QUADWORD-ATOMICS


typedef struct {
  char x[16];
} Q;

typedef _Atomic(Q) AtomicQ;

typedef __int128_t int128_t;

// PPC64-LABEL: @test_load(
// PPC64:    [[TMP3:%.*]] = load atomic i128, ptr [[TMP1:%.*]] acquire, align 16
//
Q test_load(AtomicQ *ptr) {
  // expected-no-diagnostics
  return __c11_atomic_load(ptr, __ATOMIC_ACQUIRE);
}

// PPC64-LABEL: @test_store(
// PPC64:    store atomic i128 [[TMP6:%.*]], ptr [[TMP4:%.*]] release, align 16
//
void test_store(Q val, AtomicQ *ptr) {
  // expected-no-diagnostics
  __c11_atomic_store(ptr, val, __ATOMIC_RELEASE);
}

// PPC64-LABEL: @test_add(
// PPC64:    [[ATOMICRMW:%.*]] = atomicrmw add ptr [[TMP0:%.*]], i128 [[TMP2:%.*]] monotonic, align 16
//
void test_add(_Atomic(int128_t) *ptr, int128_t x) {
  // expected-no-diagnostics
  __c11_atomic_fetch_add(ptr, x, __ATOMIC_RELAXED);
}

// PPC64-LABEL: @test_xchg(
// PPC64:    [[TMP8:%.*]] = atomicrmw xchg ptr [[TMP4:%.*]], i128 [[TMP7:%.*]] seq_cst, align 16
//
Q test_xchg(AtomicQ *ptr, Q new) {
  // expected-no-diagnostics
  return __c11_atomic_exchange(ptr, new, __ATOMIC_SEQ_CST);
}

// PPC64-LABEL: @test_cmpxchg(
// PPC64:    [[TMP10:%.*]] = cmpxchg ptr [[TMP5:%.*]], i128 [[TMP8:%.*]], i128 [[TMP9:%.*]] seq_cst monotonic, align 16
//
int test_cmpxchg(AtomicQ *ptr, Q *cmp, Q new) {
  // expected-no-diagnostics
  return __c11_atomic_compare_exchange_strong(ptr, cmp, new, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
}

// PPC64-LABEL: @test_cmpxchg_weak(
// PPC64:    [[TMP10:%.*]] = cmpxchg weak ptr [[TMP5:%.*]], i128 [[TMP8:%.*]], i128 [[TMP9:%.*]] seq_cst monotonic, align 16
//
int test_cmpxchg_weak(AtomicQ *ptr, Q *cmp, Q new) {
  // expected-no-diagnostics
  return __c11_atomic_compare_exchange_weak(ptr, cmp, new, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
}

// PPC64-QUADWORD-ATOMICS-LABEL: @is_lock_free(
// PPC64-QUADWORD-ATOMICS:    ret i32 1
//
// PPC64-NO-QUADWORD-ATOMICS-LABEL: @is_lock_free(
// PPC64-NO-QUADWORD-ATOMICS:    [[CALL:%.*]] = call zeroext i1 @__atomic_is_lock_free(i64 noundef 16, ptr noundef null)
//
int is_lock_free() {
  AtomicQ q;
 // expected-no-diagnostics
  return __c11_atomic_is_lock_free(sizeof(q));
}
