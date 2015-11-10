// RUN: %clang_cc1 %s -emit-llvm -o - -ffreestanding -triple=i686-apple-darwin9 | FileCheck %s
// REQUIRES: x86-registered-target

// Also test serialization of atomic operations here, to avoid duplicating the
// test.
// RUN: %clang_cc1 %s -emit-pch -o %t -ffreestanding -triple=i686-apple-darwin9
// RUN: %clang_cc1 %s -include-pch %t -ffreestanding -triple=i686-apple-darwin9 -emit-llvm -o - | FileCheck %s
#ifndef ALREADY_INCLUDED
#define ALREADY_INCLUDED

#include <stdatomic.h>

// Basic IRGen tests for __c11_atomic_* and GNU __atomic_*

int fi1(_Atomic(int) *i) {
  // CHECK-LABEL: @fi1
  // CHECK: load atomic i32, i32* {{.*}} seq_cst
  return __c11_atomic_load(i, memory_order_seq_cst);
}

int fi1a(int *i) {
  // CHECK-LABEL: @fi1a
  // CHECK: load atomic i32, i32* {{.*}} seq_cst
  int v;
  __atomic_load(i, &v, memory_order_seq_cst);
  return v;
}

int fi1b(int *i) {
  // CHECK-LABEL: @fi1b
  // CHECK: load atomic i32, i32* {{.*}} seq_cst
  return __atomic_load_n(i, memory_order_seq_cst);
}

int fi1c(atomic_int *i) {
  // CHECK-LABEL: @fi1c
  // CHECK: load atomic i32, i32* {{.*}} seq_cst
  return atomic_load(i);
}

void fi2(_Atomic(int) *i) {
  // CHECK-LABEL: @fi2
  // CHECK: store atomic i32 {{.*}} seq_cst
  __c11_atomic_store(i, 1, memory_order_seq_cst);
}

void fi2a(int *i) {
  // CHECK-LABEL: @fi2a
  // CHECK: store atomic i32 {{.*}} seq_cst
  int v = 1;
  __atomic_store(i, &v, memory_order_seq_cst);
}

void fi2b(int *i) {
  // CHECK-LABEL: @fi2b
  // CHECK: store atomic i32 {{.*}} seq_cst
  __atomic_store_n(i, 1, memory_order_seq_cst);
}

void fi2c(atomic_int *i) {
  // CHECK-LABEL: @fi2c
  // CHECK: store atomic i32 {{.*}} seq_cst
  atomic_store(i, 1);
}

int fi3(_Atomic(int) *i) {
  // CHECK-LABEL: @fi3
  // CHECK: atomicrmw and
  // CHECK-NOT: and
  return __c11_atomic_fetch_and(i, 1, memory_order_seq_cst);
}

int fi3a(int *i) {
  // CHECK-LABEL: @fi3a
  // CHECK: atomicrmw xor
  // CHECK-NOT: xor
  return __atomic_fetch_xor(i, 1, memory_order_seq_cst);
}

int fi3b(int *i) {
  // CHECK-LABEL: @fi3b
  // CHECK: atomicrmw add
  // CHECK: add
  return __atomic_add_fetch(i, 1, memory_order_seq_cst);
}

int fi3c(int *i) {
  // CHECK-LABEL: @fi3c
  // CHECK: atomicrmw nand
  // CHECK-NOT: and
  return __atomic_fetch_nand(i, 1, memory_order_seq_cst);
}

int fi3d(int *i) {
  // CHECK-LABEL: @fi3d
  // CHECK: atomicrmw nand
  // CHECK: and
  // CHECK: xor
  return __atomic_nand_fetch(i, 1, memory_order_seq_cst);
}

int fi3e(atomic_int *i) {
  // CHECK-LABEL: @fi3e
  // CHECK: atomicrmw or
  // CHECK-NOT: {{ or }}
  return atomic_fetch_or(i, 1);
}

int fi3f(int *i) {
  // CHECK-LABEL: @fi3f
  // CHECK-NOT: store volatile
  // CHECK: atomicrmw or
  // CHECK-NOT: {{ or }}
  return __atomic_fetch_or(i, (short)1, memory_order_seq_cst);
}

_Bool fi4(_Atomic(int) *i) {
  // CHECK-LABEL: @fi4(
  // CHECK: [[PAIR:%[.0-9A-Z_a-z]+]] = cmpxchg i32* [[PTR:%[.0-9A-Z_a-z]+]], i32 [[EXPECTED:%[.0-9A-Z_a-z]+]], i32 [[DESIRED:%[.0-9A-Z_a-z]+]]
  // CHECK: [[OLD:%[.0-9A-Z_a-z]+]] = extractvalue { i32, i1 } [[PAIR]], 0
  // CHECK: [[CMP:%[.0-9A-Z_a-z]+]] = extractvalue { i32, i1 } [[PAIR]], 1
  // CHECK: br i1 [[CMP]], label %[[STORE_EXPECTED:[.0-9A-Z_a-z]+]], label %[[CONTINUE:[.0-9A-Z_a-z]+]]
  // CHECK: store i32 [[OLD]]
  int cmp = 0;
  return __c11_atomic_compare_exchange_strong(i, &cmp, 1, memory_order_acquire, memory_order_acquire);
}

_Bool fi4a(int *i) {
  // CHECK-LABEL: @fi4a
  // CHECK: [[PAIR:%[.0-9A-Z_a-z]+]] = cmpxchg i32* [[PTR:%[.0-9A-Z_a-z]+]], i32 [[EXPECTED:%[.0-9A-Z_a-z]+]], i32 [[DESIRED:%[.0-9A-Z_a-z]+]]
  // CHECK: [[OLD:%[.0-9A-Z_a-z]+]] = extractvalue { i32, i1 } [[PAIR]], 0
  // CHECK: [[CMP:%[.0-9A-Z_a-z]+]] = extractvalue { i32, i1 } [[PAIR]], 1
  // CHECK: br i1 [[CMP]], label %[[STORE_EXPECTED:[.0-9A-Z_a-z]+]], label %[[CONTINUE:[.0-9A-Z_a-z]+]]
  // CHECK: store i32 [[OLD]]
  int cmp = 0;
  int desired = 1;
  return __atomic_compare_exchange(i, &cmp, &desired, 0, memory_order_acquire, memory_order_acquire);
}

_Bool fi4b(int *i) {
  // CHECK-LABEL: @fi4b(
  // CHECK: [[PAIR:%[.0-9A-Z_a-z]+]] = cmpxchg weak i32* [[PTR:%[.0-9A-Z_a-z]+]], i32 [[EXPECTED:%[.0-9A-Z_a-z]+]], i32 [[DESIRED:%[.0-9A-Z_a-z]+]]
  // CHECK: [[OLD:%[.0-9A-Z_a-z]+]] = extractvalue { i32, i1 } [[PAIR]], 0
  // CHECK: [[CMP:%[.0-9A-Z_a-z]+]] = extractvalue { i32, i1 } [[PAIR]], 1
  // CHECK: br i1 [[CMP]], label %[[STORE_EXPECTED:[.0-9A-Z_a-z]+]], label %[[CONTINUE:[.0-9A-Z_a-z]+]]
  // CHECK: store i32 [[OLD]]
  int cmp = 0;
  return __atomic_compare_exchange_n(i, &cmp, 1, 1, memory_order_acquire, memory_order_acquire);
}

_Bool fi4c(atomic_int *i) {
  // CHECK-LABEL: @fi4c
  // CHECK: cmpxchg i32*
  int cmp = 0;
  return atomic_compare_exchange_strong(i, &cmp, 1);
}

float ff1(_Atomic(float) *d) {
  // CHECK-LABEL: @ff1
  // CHECK: load atomic i32, i32* {{.*}} monotonic
  return __c11_atomic_load(d, memory_order_relaxed);
}

void ff2(_Atomic(float) *d) {
  // CHECK-LABEL: @ff2
  // CHECK: store atomic i32 {{.*}} release
  __c11_atomic_store(d, 1, memory_order_release);
}

float ff3(_Atomic(float) *d) {
  return __c11_atomic_exchange(d, 2, memory_order_seq_cst);
}

struct S {
  double x;
};

struct S fd1(struct S *a) {
  // CHECK-LABEL: @fd1
  // CHECK: [[RETVAL:%.*]] = alloca %struct.S, align 4
  // CHECK: [[RET:%.*]]    = alloca %struct.S, align 4
  // CHECK: [[CAST:%.*]]   = bitcast %struct.S* [[RET]] to i64*
  // CHECK: [[CALL:%.*]]   = call i64 @__atomic_load_8(
  // CHECK: store i64 [[CALL]], i64* [[CAST]], align 4
  struct S ret;
  __atomic_load(a, &ret, memory_order_seq_cst);
  return ret;
}

void fd2(struct S *a, struct S *b) {
  // CHECK-LABEL: @fd2
  // CHECK:      [[A_ADDR:%.*]] = alloca %struct.S*, align 4
  // CHECK-NEXT: [[B_ADDR:%.*]] = alloca %struct.S*, align 4
  // CHECK-NEXT: store %struct.S* %a, %struct.S** [[A_ADDR]], align 4
  // CHECK-NEXT: store %struct.S* %b, %struct.S** [[B_ADDR]], align 4
  // CHECK-NEXT: [[LOAD_A_PTR:%.*]] = load %struct.S*, %struct.S** [[A_ADDR]], align 4
  // CHECK-NEXT: [[LOAD_B_PTR:%.*]] = load %struct.S*, %struct.S** [[B_ADDR]], align 4
  // CHECK-NEXT: [[COERCED_A_TMP:%.*]] = bitcast %struct.S* [[LOAD_A_PTR]] to i64*
  // CHECK-NEXT: [[COERCED_B:%.*]] = bitcast %struct.S* [[LOAD_B_PTR]] to i64*
  // CHECK-NEXT: [[COERCED_A:%.*]] = bitcast i64* [[COERCED_A_TMP]] to i8*
  // CHECK-NEXT: [[LOAD_B:%.*]] = load i64, i64* [[COERCED_B]], align 4
  // CHECK-NEXT: call void @__atomic_store_8(i8* [[COERCED_A]], i64 [[LOAD_B]],
  // CHECK-NEXT: ret void
  __atomic_store(a, b, memory_order_seq_cst);
}

void fd3(struct S *a, struct S *b, struct S *c) {
  // CHECK-LABEL: @fd3
  // CHECK:      [[A_ADDR:%.*]] = alloca %struct.S*, align 4
  // CHECK-NEXT: [[B_ADDR:%.*]] = alloca %struct.S*, align 4
  // CHECK-NEXT: [[C_ADDR:%.*]] = alloca %struct.S*, align 4
  // CHECK-NEXT: store %struct.S* %a, %struct.S** [[A_ADDR]], align 4
  // CHECK-NEXT: store %struct.S* %b, %struct.S** [[B_ADDR]], align 4
  // CHECK-NEXT: store %struct.S* %c, %struct.S** [[C_ADDR]], align 4
  // CHECK-NEXT: [[LOAD_A_PTR:%.*]] = load %struct.S*, %struct.S** [[A_ADDR]], align 4
  // CHECK-NEXT: [[LOAD_B_PTR:%.*]] = load %struct.S*, %struct.S** [[B_ADDR]], align 4
  // CHECK-NEXT: [[LOAD_C_PTR:%.*]] = load %struct.S*, %struct.S** [[C_ADDR]], align 4
  // CHECK-NEXT: [[COERCED_A_TMP:%.*]] = bitcast %struct.S* [[LOAD_A_PTR]] to i64*
  // CHECK-NEXT: [[COERCED_B:%.*]] = bitcast %struct.S* [[LOAD_B_PTR]] to i64*
  // CHECK-NEXT: [[COERCED_C:%.*]] = bitcast %struct.S* [[LOAD_C_PTR]] to i64*
  // CHECK-NEXT: [[COERCED_A:%.*]] = bitcast i64* [[COERCED_A_TMP]] to i8*
  // CHECK-NEXT: [[LOAD_B:%.*]] = load i64, i64* [[COERCED_B]], align 4
  // CHECK-NEXT: [[CALL:%.*]] = call i64 @__atomic_exchange_8(i8* [[COERCED_A]], i64 [[LOAD_B]],
  // CHECK-NEXT: store i64 [[CALL]], i64* [[COERCED_C]], align 4

  __atomic_exchange(a, b, c, memory_order_seq_cst);
}

_Bool fd4(struct S *a, struct S *b, struct S *c) {
  // CHECK-LABEL: @fd4
  // CHECK:      [[A_ADDR:%.*]] = alloca %struct.S*, align 4
  // CHECK-NEXT: [[B_ADDR:%.*]] = alloca %struct.S*, align 4
  // CHECK-NEXT: [[C_ADDR:%.*]] = alloca %struct.S*, align 4
  // CHECK:      store %struct.S* %a, %struct.S** [[A_ADDR]], align 4
  // CHECK-NEXT: store %struct.S* %b, %struct.S** [[B_ADDR]], align 4
  // CHECK-NEXT: store %struct.S* %c, %struct.S** [[C_ADDR]], align 4
  // CHECK-NEXT: [[LOAD_A_PTR:%.*]] = load %struct.S*, %struct.S** [[A_ADDR]], align 4
  // CHECK-NEXT: [[LOAD_B_PTR:%.*]] = load %struct.S*, %struct.S** [[B_ADDR]], align 4
  // CHECK-NEXT: [[LOAD_C_PTR:%.*]] = load %struct.S*, %struct.S** [[C_ADDR]], align 4
  // CHECK-NEXT: [[COERCED_A_TMP:%.*]] = bitcast %struct.S* [[LOAD_A_PTR]] to i64*
  // CHECK-NEXT: [[COERCED_B_TMP:%.*]] = bitcast %struct.S* [[LOAD_B_PTR]] to i64*
  // CHECK-NEXT: [[COERCED_C:%.*]] = bitcast %struct.S* [[LOAD_C_PTR]] to i64*
  // CHECK-NEXT: [[COERCED_A:%.*]] = bitcast i64* [[COERCED_A_TMP]] to i8*
  // CHECK-NEXT: [[COERCED_B:%.*]] = bitcast i64* [[COERCED_B_TMP]] to i8*
  // CHECK-NEXT: [[LOAD_C:%.*]] = load i64, i64* [[COERCED_C]], align 4
  // CHECK-NEXT: [[CALL:%.*]] = call zeroext i1 @__atomic_compare_exchange_8(i8* [[COERCED_A]], i8* [[COERCED_B]], i64 [[LOAD_C]]
  // CHECK-NEXT: ret i1 [[CALL]]
  return __atomic_compare_exchange(a, b, c, 1, 5, 5);
}

int* fp1(_Atomic(int*) *p) {
  // CHECK-LABEL: @fp1
  // CHECK: load atomic i32, i32* {{.*}} seq_cst
  return __c11_atomic_load(p, memory_order_seq_cst);
}

int* fp2(_Atomic(int*) *p) {
  // CHECK-LABEL: @fp2
  // CHECK: store i32 4
  // CHECK: atomicrmw add {{.*}} monotonic
  return __c11_atomic_fetch_add(p, 1, memory_order_relaxed);
}

int *fp2a(int **p) {
  // CHECK-LABEL: @fp2a
  // CHECK: store i32 4
  // CHECK: atomicrmw sub {{.*}} monotonic
  // Note, the GNU builtins do not multiply by sizeof(T)!
  return __atomic_fetch_sub(p, 4, memory_order_relaxed);
}

_Complex float fc(_Atomic(_Complex float) *c) {
  // CHECK-LABEL: @fc
  // CHECK: atomicrmw xchg i64*
  return __c11_atomic_exchange(c, 2, memory_order_seq_cst);
}

typedef struct X { int x; } X;
X fs(_Atomic(X) *c) {
  // CHECK-LABEL: @fs
  // CHECK: atomicrmw xchg i32*
  return __c11_atomic_exchange(c, (X){2}, memory_order_seq_cst);
}

X fsa(X *c, X *d) {
  // CHECK-LABEL: @fsa
  // CHECK: atomicrmw xchg i32*
  X ret;
  __atomic_exchange(c, d, &ret, memory_order_seq_cst);
  return ret;
}

_Bool fsb(_Bool *c) {
  // CHECK-LABEL: @fsb
  // CHECK: atomicrmw xchg i8*
  return __atomic_exchange_n(c, 1, memory_order_seq_cst);
}

char flag1;
volatile char flag2;
void test_and_set() {
  // CHECK: atomicrmw xchg i8* @flag1, i8 1 seq_cst
  __atomic_test_and_set(&flag1, memory_order_seq_cst);
  // CHECK: atomicrmw volatile xchg i8* @flag2, i8 1 acquire
  __atomic_test_and_set(&flag2, memory_order_acquire);
  // CHECK: store atomic volatile i8 0, i8* @flag2 release
  __atomic_clear(&flag2, memory_order_release);
  // CHECK: store atomic i8 0, i8* @flag1 seq_cst
  __atomic_clear(&flag1, memory_order_seq_cst);
}

struct Sixteen {
  char c[16];
} sixteen;
struct Seventeen {
  char c[17];
} seventeen;

struct Incomplete;

int lock_free(struct Incomplete *incomplete) {
  // CHECK-LABEL: @lock_free

  // CHECK: call i32 @__atomic_is_lock_free(i32 3, i8* null)
  __c11_atomic_is_lock_free(3);

  // CHECK: call i32 @__atomic_is_lock_free(i32 16, i8* {{.*}}@sixteen{{.*}})
  __atomic_is_lock_free(16, &sixteen);

  // CHECK: call i32 @__atomic_is_lock_free(i32 17, i8* {{.*}}@seventeen{{.*}})
  __atomic_is_lock_free(17, &seventeen);

  // CHECK: call i32 @__atomic_is_lock_free(i32 4, {{.*}})
  __atomic_is_lock_free(4, incomplete);

  char cs[20];
  // CHECK: call i32 @__atomic_is_lock_free(i32 4, {{.*}})
  __atomic_is_lock_free(4, cs+1);

  // CHECK-NOT: call
  __atomic_always_lock_free(3, 0);
  __atomic_always_lock_free(16, 0);
  __atomic_always_lock_free(17, 0);
  __atomic_always_lock_free(16, &sixteen);
  __atomic_always_lock_free(17, &seventeen);

  int n;
  __atomic_is_lock_free(4, &n);

  // CHECK: ret i32 1
  return __c11_atomic_is_lock_free(sizeof(_Atomic(int)));
}

// Tests for atomic operations on big values.  These should call the functions
// defined here:
// http://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary#The_Library_interface

struct foo {
  int big[128];
};
struct bar {
  char c[3];
};

struct bar smallThing, thing1, thing2;
struct foo bigThing;
_Atomic(struct foo) bigAtomic;

void structAtomicStore() {
  // CHECK-LABEL: @structAtomicStore
  struct foo f = {0};
  struct bar b = {0};
  __atomic_store(&smallThing, &b, 5);
  // CHECK: call void @__atomic_store(i32 3, i8* {{.*}} @smallThing

  __atomic_store(&bigThing, &f, 5);
  // CHECK: call void @__atomic_store(i32 512, i8* {{.*}} @bigThing
}
void structAtomicLoad() {
  // CHECK-LABEL: @structAtomicLoad
  struct bar b;
  __atomic_load(&smallThing, &b, 5);
  // CHECK: call void @__atomic_load(i32 3, i8* {{.*}} @smallThing

  struct foo f = {0};
  __atomic_load(&bigThing, &f, 5);
  // CHECK: call void @__atomic_load(i32 512, i8* {{.*}} @bigThing
}
struct foo structAtomicExchange() {
  // CHECK-LABEL: @structAtomicExchange
  struct foo f = {0};
  struct foo old;
  __atomic_exchange(&f, &bigThing, &old, 5);
  // CHECK: call void @__atomic_exchange(i32 512, {{.*}}, i8* bitcast ({{.*}} @bigThing to i8*),

  return __c11_atomic_exchange(&bigAtomic, f, 5);
  // CHECK: call void @__atomic_exchange(i32 512, i8* bitcast ({{.*}} @bigAtomic to i8*),
}
int structAtomicCmpExchange() {
  // CHECK-LABEL: @structAtomicCmpExchange
  // CHECK: %[[x_mem:.*]] = alloca i8
  _Bool x = __atomic_compare_exchange(&smallThing, &thing1, &thing2, 1, 5, 5);
  // CHECK: %[[call1:.*]] = call zeroext i1 @__atomic_compare_exchange(i32 3, {{.*}} @smallThing{{.*}} @thing1{{.*}} @thing2
  // CHECK: %[[zext1:.*]] = zext i1 %[[call1]] to i8
  // CHECK: store i8 %[[zext1]], i8* %[[x_mem]], align 1
  // CHECK: %[[x:.*]] = load i8, i8* %[[x_mem]]
  // CHECK: %[[x_bool:.*]] = trunc i8 %[[x]] to i1
  // CHECK: %[[conv1:.*]] = zext i1 %[[x_bool]] to i32

  struct foo f = {0};
  struct foo g = {0};
  g.big[12] = 12;
  return x & __c11_atomic_compare_exchange_strong(&bigAtomic, &f, g, 5, 5);
  // CHECK: %[[call2:.*]] = call zeroext i1 @__atomic_compare_exchange(i32 512, i8* bitcast ({{.*}} @bigAtomic to i8*),
  // CHECK: %[[conv2:.*]] = zext i1 %[[call2]] to i32
  // CHECK: %[[and:.*]] = and i32 %[[conv1]], %[[conv2]]
  // CHECK: ret i32 %[[and]]
}

// Check that no atomic operations are used in any initialisation of _Atomic
// types.
_Atomic(int) atomic_init_i = 42;

// CHECK-LABEL: @atomic_init_foo
void atomic_init_foo()
{
  // CHECK-NOT: }
  // CHECK-NOT: atomic
  // CHECK: store
  _Atomic(int) j = 12;

  // CHECK-NOT: }
  // CHECK-NOT: atomic
  // CHECK: store
  __c11_atomic_init(&j, 42);

  // CHECK-NOT: atomic
  // CHECK: }
}

// CHECK-LABEL: @failureOrder
void failureOrder(_Atomic(int) *ptr, int *ptr2) {
  __c11_atomic_compare_exchange_strong(ptr, ptr2, 43, memory_order_acquire, memory_order_relaxed);
  // CHECK: cmpxchg i32* {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z_.]+}} acquire monotonic

  __c11_atomic_compare_exchange_weak(ptr, ptr2, 43, memory_order_seq_cst, memory_order_acquire);
  // CHECK: cmpxchg weak i32* {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z_.]+}} seq_cst acquire

  // Unknown ordering: conservatively pick strongest valid option (for now!).
  __atomic_compare_exchange(ptr2, ptr2, ptr2, 0, memory_order_acq_rel, *ptr2);
  // CHECK: cmpxchg i32* {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z_.]+}} acq_rel acquire

  // Undefined behaviour: don't really care what that last ordering is so leave
  // it out:
  __atomic_compare_exchange_n(ptr2, ptr2, 43, 1, memory_order_seq_cst, 42);
  // CHECK: cmpxchg weak i32* {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z_.]+}} seq_cst
}

// CHECK-LABEL: @generalFailureOrder
void generalFailureOrder(_Atomic(int) *ptr, int *ptr2, int success, int fail) {
  __c11_atomic_compare_exchange_strong(ptr, ptr2, 42, success, fail);
  // CHECK: switch i32 {{.*}}, label %[[MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[ACQUIRE]]
  // CHECK-NEXT: i32 3, label %[[RELEASE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 4, label %[[ACQREL:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 5, label %[[SEQCST:[0-9a-zA-Z._]+]]

  // CHECK: [[MONOTONIC]]
  // CHECK: switch {{.*}}, label %[[MONOTONIC_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: ]

  // CHECK: [[ACQUIRE]]
  // CHECK: switch {{.*}}, label %[[ACQUIRE_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[ACQUIRE_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[ACQUIRE_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: ]

  // CHECK: [[RELEASE]]
  // CHECK: switch {{.*}}, label %[[RELEASE_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: ]

  // CHECK: [[ACQREL]]
  // CHECK: switch {{.*}}, label %[[ACQREL_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[ACQREL_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[ACQREL_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: ]

  // CHECK: [[SEQCST]]
  // CHECK: switch {{.*}}, label %[[SEQCST_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[SEQCST_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[SEQCST_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 5, label %[[SEQCST_SEQCST:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: ]

  // CHECK: [[MONOTONIC_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} monotonic monotonic
  // CHECK: br

  // CHECK: [[ACQUIRE_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} acquire monotonic
  // CHECK: br

  // CHECK: [[ACQUIRE_ACQUIRE]]
  // CHECK: cmpxchg {{.*}} acquire acquire
  // CHECK: br

  // CHECK: [[ACQREL_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} acq_rel monotonic
  // CHECK: br

  // CHECK: [[ACQREL_ACQUIRE]]
  // CHECK: cmpxchg {{.*}} acq_rel acquire
  // CHECK: br

  // CHECK: [[SEQCST_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} seq_cst monotonic
  // CHECK: br

  // CHECK: [[SEQCST_ACQUIRE]]
  // CHECK: cmpxchg {{.*}} seq_cst acquire
  // CHECK: br

  // CHECK: [[SEQCST_SEQCST]]
  // CHECK: cmpxchg {{.*}} seq_cst seq_cst
  // CHECK: br
}

void generalWeakness(int *ptr, int *ptr2, _Bool weak) {
  __atomic_compare_exchange_n(ptr, ptr2, 42, weak, memory_order_seq_cst, memory_order_seq_cst);
  // CHECK: switch i1 {{.*}}, label %[[WEAK:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i1 false, label %[[STRONG:[0-9a-zA-Z._]+]]

  // CHECK: [[STRONG]]
  // CHECK-NOT: br
  // CHECK: cmpxchg {{.*}} seq_cst seq_cst
  // CHECK: br

  // CHECK: [[WEAK]]
  // CHECK-NOT: br
  // CHECK: cmpxchg weak {{.*}} seq_cst seq_cst
  // CHECK: br
}

// Having checked the flow in the previous two cases, we'll trust clang to
// combine them sanely.
void EMIT_ALL_THE_THINGS(int *ptr, int *ptr2, int new, _Bool weak, int success, int fail) {
  __atomic_compare_exchange(ptr, ptr2, &new, weak, success, fail);

  // CHECK: = cmpxchg {{.*}} monotonic monotonic
  // CHECK: = cmpxchg weak {{.*}} monotonic monotonic
  // CHECK: = cmpxchg {{.*}} acquire monotonic
  // CHECK: = cmpxchg {{.*}} acquire acquire
  // CHECK: = cmpxchg weak {{.*}} acquire monotonic
  // CHECK: = cmpxchg weak {{.*}} acquire acquire
  // CHECK: = cmpxchg {{.*}} release monotonic
  // CHECK: = cmpxchg weak {{.*}} release monotonic
  // CHECK: = cmpxchg {{.*}} acq_rel monotonic
  // CHECK: = cmpxchg {{.*}} acq_rel acquire
  // CHECK: = cmpxchg weak {{.*}} acq_rel monotonic
  // CHECK: = cmpxchg weak {{.*}} acq_rel acquire
  // CHECK: = cmpxchg {{.*}} seq_cst monotonic
  // CHECK: = cmpxchg {{.*}} seq_cst acquire
  // CHECK: = cmpxchg {{.*}} seq_cst seq_cst
  // CHECK: = cmpxchg weak {{.*}} seq_cst monotonic
  // CHECK: = cmpxchg weak {{.*}} seq_cst acquire
  // CHECK: = cmpxchg weak {{.*}} seq_cst seq_cst
}

int PR21643() {
  return __atomic_or_fetch((int __attribute__((address_space(257))) *)0x308, 1,
                           __ATOMIC_RELAXED);
  // CHECK: %[[atomictmp:.*]] = alloca i32, align 4
  // CHECK: %[[atomicdst:.*]] = alloca i32, align 4
  // CHECK: store i32 1, i32* %[[atomictmp]]
  // CHECK: %[[one:.*]] = load i32, i32* %[[atomictmp]], align 4
  // CHECK: %[[old:.*]] = atomicrmw or i32 addrspace(257)* inttoptr (i32 776 to i32 addrspace(257)*), i32 %[[one]] monotonic
  // CHECK: %[[new:.*]] = or i32 %[[old]], %[[one]]
  // CHECK: store i32 %[[new]], i32* %[[atomicdst]], align 4
  // CHECK: %[[ret:.*]] = load i32, i32* %[[atomicdst]], align 4
  // CHECK: ret i32 %[[ret]]
}

int PR17306_1(volatile _Atomic(int) *i) {
  // CHECK-LABEL: @PR17306_1
  // CHECK:      %[[i_addr:.*]] = alloca i32
  // CHECK-NEXT: %[[atomicdst:.*]] = alloca i32
  // CHECK-NEXT: store i32* %i, i32** %[[i_addr]]
  // CHECK-NEXT: %[[addr:.*]] = load i32*, i32** %[[i_addr]]
  // CHECK-NEXT: %[[res:.*]] = load atomic volatile i32, i32* %[[addr]] seq_cst
  // CHECK-NEXT: store i32 %[[res]], i32* %[[atomicdst]]
  // CHECK-NEXT: %[[retval:.*]] = load i32, i32* %[[atomicdst]]
  // CHECK-NEXT: ret i32 %[[retval]]
  return __c11_atomic_load(i, memory_order_seq_cst);
}

int PR17306_2(volatile int *i, int value) {
  // CHECK-LABEL: @PR17306_2
  // CHECK:      %[[i_addr:.*]] = alloca i32*
  // CHECK-NEXT: %[[value_addr:.*]] = alloca i32
  // CHECK-NEXT: %[[atomictmp:.*]] = alloca i32
  // CHECK-NEXT: %[[atomicdst:.*]] = alloca i32
  // CHECK-NEXT: store i32* %i, i32** %[[i_addr]]
  // CHECK-NEXT: store i32 %value, i32* %[[value_addr]]
  // CHECK-NEXT: %[[i_lval:.*]] = load i32*, i32** %[[i_addr]]
  // CHECK-NEXT: %[[value:.*]] = load i32, i32* %[[value_addr]]
  // CHECK-NEXT: store i32 %[[value]], i32* %[[atomictmp]]
  // CHECK-NEXT: %[[value_lval:.*]] = load i32, i32* %[[atomictmp]]
  // CHECK-NEXT: %[[old_val:.*]] = atomicrmw volatile add i32* %[[i_lval]], i32 %[[value_lval]] seq_cst
  // CHECK-NEXT: %[[new_val:.*]] = add i32 %[[old_val]], %[[value_lval]]
  // CHECK-NEXT: store i32 %[[new_val]], i32* %[[atomicdst]]
  // CHECK-NEXT: %[[retval:.*]] = load i32, i32* %[[atomicdst]]
  // CHECK-NEXT: ret i32 %[[retval]]
  return __atomic_add_fetch(i, value, memory_order_seq_cst);
}

#endif
