// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-macosx10.9.0 | FileCheck %s
// REQUIRES: x86-registered-target
// Also test serialization of atomic operations here, to avoid duplicating the
// test.
// RUN: %clang_cc1 %s -emit-pch -o %t -triple=x86_64-apple-macosx10.9.0
// RUN: %clang_cc1 %s -include-pch %t -triple=x86_64-apple-macosx10.9.0 -emit-llvm -o - | FileCheck %s
#ifndef ALREADY_INCLUDED
#define ALREADY_INCLUDED

// Basic IRGen tests for __c11_atomic_* and GNU __atomic_*

typedef enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
} memory_order;

int fi1(_Atomic(int) *i) {
  // CHECK: @fi1
  // CHECK: load atomic i32, i32* {{.*}} seq_cst, align 4
  return __c11_atomic_load(i, memory_order_seq_cst);
}

int fi1a(int *i) {
  // CHECK: @fi1a
  // CHECK: load atomic i32, i32* {{.*}} seq_cst, align 4
  int v;
  __atomic_load(i, &v, memory_order_seq_cst);
  return v;
}

int fi1b(int *i) {
  // CHECK: @fi1b
  // CHECK: load atomic i32, i32* {{.*}} seq_cst, align 4
  return __atomic_load_n(i, memory_order_seq_cst);
}

void fi2(_Atomic(int) *i) {
  // CHECK: @fi2
  // CHECK: store atomic i32 {{.*}} seq_cst, align 4
  __c11_atomic_store(i, 1, memory_order_seq_cst);
}

void fi2a(int *i) {
  // CHECK: @fi2a
  // CHECK: store atomic i32 {{.*}} seq_cst, align 4
  int v = 1;
  __atomic_store(i, &v, memory_order_seq_cst);
}

void fi2b(int *i) {
  // CHECK: @fi2b
  // CHECK: store atomic i32 {{.*}} seq_cst, align 4
  __atomic_store_n(i, 1, memory_order_seq_cst);
}

int fi3(_Atomic(int) *i) {
  // CHECK: @fi3
  // CHECK: atomicrmw and {{.*}} seq_cst, align 4
  // CHECK-NOT: and
  return __c11_atomic_fetch_and(i, 1, memory_order_seq_cst);
}

int fi3a(int *i) {
  // CHECK: @fi3a
  // CHECK: atomicrmw xor {{.*}} seq_cst, align 4
  // CHECK-NOT: xor
  return __atomic_fetch_xor(i, 1, memory_order_seq_cst);
}

int fi3b(int *i) {
  // CHECK: @fi3b
  // CHECK: atomicrmw add {{.*}} seq_cst, align 4
  // CHECK: add
  return __atomic_add_fetch(i, 1, memory_order_seq_cst);
}

int fi3c(int *i) {
  // CHECK: @fi3c
  // CHECK: atomicrmw nand {{.*}} seq_cst, align 4
  // CHECK-NOT: and
  return __atomic_fetch_nand(i, 1, memory_order_seq_cst);
}

int fi3d(int *i) {
  // CHECK: @fi3d
  // CHECK: atomicrmw nand {{.*}} seq_cst, align 4
  // CHECK: and
  // CHECK: xor
  return __atomic_nand_fetch(i, 1, memory_order_seq_cst);
}

_Bool fi4(_Atomic(int) *i) {
  // CHECK: @fi4
  // CHECK: cmpxchg i32* {{.*}} acquire acquire, align 4
  int cmp = 0;
  return __c11_atomic_compare_exchange_strong(i, &cmp, 1, memory_order_acquire, memory_order_acquire);
}

_Bool fi4a(int *i) {
  // CHECK: @fi4
  // CHECK: cmpxchg i32* {{.*}} acquire acquire, align 4
  int cmp = 0;
  int desired = 1;
  return __atomic_compare_exchange(i, &cmp, &desired, 0, memory_order_acquire, memory_order_acquire);
}

_Bool fi4b(int *i) {
  // CHECK: @fi4
  // CHECK: cmpxchg weak i32* {{.*}} acquire acquire, align 4
  int cmp = 0;
  return __atomic_compare_exchange_n(i, &cmp, 1, 1, memory_order_acquire, memory_order_acquire);
}

float ff1(_Atomic(float) *d) {
  // CHECK: @ff1
  // CHECK: load atomic i32, i32* {{.*}} monotonic, align 4
  return __c11_atomic_load(d, memory_order_relaxed);
}

void ff2(_Atomic(float) *d) {
  // CHECK: @ff2
  // CHECK: store atomic i32 {{.*}} release, align 4
  __c11_atomic_store(d, 1, memory_order_release);
}

float ff3(_Atomic(float) *d) {
  return __c11_atomic_exchange(d, 2, memory_order_seq_cst);
}

int* fp1(_Atomic(int*) *p) {
  // CHECK: @fp1
  // CHECK: load atomic i64, i64* {{.*}} seq_cst, align 8
  return __c11_atomic_load(p, memory_order_seq_cst);
}

int* fp2(_Atomic(int*) *p) {
  // CHECK: @fp2
  // CHECK: store i64 4
  // CHECK: atomicrmw add {{.*}} monotonic, align 8
  return __c11_atomic_fetch_add(p, 1, memory_order_relaxed);
}

int *fp2a(int **p) {
  // CHECK: @fp2a
  // CHECK: store i64 4
  // CHECK: atomicrmw sub {{.*}} monotonic, align 8
  // Note, the GNU builtins do not multiply by sizeof(T)!
  return __atomic_fetch_sub(p, 4, memory_order_relaxed);
}

_Complex float fc(_Atomic(_Complex float) *c) {
  // CHECK: @fc
  // CHECK: atomicrmw xchg i64* {{.*}} seq_cst, align 8
  return __c11_atomic_exchange(c, 2, memory_order_seq_cst);
}

typedef struct X { int x; } X;
X fs(_Atomic(X) *c) {
  // CHECK: @fs
  // CHECK: atomicrmw xchg i32* {{.*}} seq_cst, align 4
  return __c11_atomic_exchange(c, (X){2}, memory_order_seq_cst);
}

X fsa(X *c, X *d) {
  // CHECK: @fsa
  // CHECK: atomicrmw xchg i32* {{.*}} seq_cst, align 4
  X ret;
  __atomic_exchange(c, d, &ret, memory_order_seq_cst);
  return ret;
}

_Bool fsb(_Bool *c) {
  // CHECK: @fsb
  // CHECK: atomicrmw xchg i8* {{.*}} seq_cst, align 1
  return __atomic_exchange_n(c, 1, memory_order_seq_cst);
}

char flag1;
volatile char flag2;
void test_and_set() {
  // CHECK: atomicrmw xchg i8* @flag1, i8 1 seq_cst, align 1
  __atomic_test_and_set(&flag1, memory_order_seq_cst);
  // CHECK: atomicrmw volatile xchg i8* @flag2, i8 1 acquire, align 1
  __atomic_test_and_set(&flag2, memory_order_acquire);
  // CHECK: store atomic volatile i8 0, i8* @flag2 release, align 1
  __atomic_clear(&flag2, memory_order_release);
  // CHECK: store atomic i8 0, i8* @flag1 seq_cst, align 1
  __atomic_clear(&flag1, memory_order_seq_cst);
}

struct Sixteen {
  char c[16];
} sixteen;
struct Seventeen {
  char c[17];
} seventeen;

int lock_free(struct Incomplete *incomplete) {
  // CHECK: @lock_free

  // CHECK: call zeroext i1 @__atomic_is_lock_free(i64 noundef 3, i8* noundef null)
  __c11_atomic_is_lock_free(3);

  // CHECK: call zeroext i1 @__atomic_is_lock_free(i64 noundef 16, i8* noundef {{.*}}@sixteen{{.*}})
  __atomic_is_lock_free(16, &sixteen);

  // CHECK: call zeroext i1 @__atomic_is_lock_free(i64 noundef 17, i8* noundef {{.*}}@seventeen{{.*}})
  __atomic_is_lock_free(17, &seventeen);

  // CHECK: call zeroext i1 @__atomic_is_lock_free(i64 noundef 4, {{.*}})
  __atomic_is_lock_free(4, incomplete);

  char cs[20];
  // CHECK: call zeroext i1 @__atomic_is_lock_free(i64 noundef 4, {{.*}})
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
  // CHECK: @structAtomicStore
  struct foo f = {0};
  __c11_atomic_store(&bigAtomic, f, 5);
  // CHECK: call void @__atomic_store(i64 noundef 512, i8* noundef bitcast ({{.*}} @bigAtomic to i8*),

  struct bar b = {0};
  __atomic_store(&smallThing, &b, 5);
  // CHECK: call void @__atomic_store(i64 noundef 3, i8* noundef {{.*}} @smallThing

  __atomic_store(&bigThing, &f, 5);
  // CHECK: call void @__atomic_store(i64 noundef 512, i8* noundef {{.*}} @bigThing
}
void structAtomicLoad() {
  // CHECK: @structAtomicLoad
  struct foo f = __c11_atomic_load(&bigAtomic, 5);
  // CHECK: call void @__atomic_load(i64 noundef 512, i8* noundef bitcast ({{.*}} @bigAtomic to i8*),

  struct bar b;
  __atomic_load(&smallThing, &b, 5);
  // CHECK: call void @__atomic_load(i64 noundef 3, i8* noundef {{.*}} @smallThing

  __atomic_load(&bigThing, &f, 5);
  // CHECK: call void @__atomic_load(i64 noundef 512, i8* noundef {{.*}} @bigThing
}
struct foo structAtomicExchange() {
  // CHECK: @structAtomicExchange
  struct foo f = {0};
  struct foo old;
  __atomic_exchange(&f, &bigThing, &old, 5);
  // CHECK: call void @__atomic_exchange(i64 noundef 512, {{.*}}, i8* noundef bitcast ({{.*}} @bigThing to i8*),

  return __c11_atomic_exchange(&bigAtomic, f, 5);
  // CHECK: call void @__atomic_exchange(i64 noundef 512, i8* noundef bitcast ({{.*}} @bigAtomic to i8*),
}
int structAtomicCmpExchange() {
  // CHECK: @structAtomicCmpExchange
  _Bool x = __atomic_compare_exchange(&smallThing, &thing1, &thing2, 1, 5, 5);
  // CHECK: call zeroext i1 @__atomic_compare_exchange(i64 noundef 3, {{.*}} @smallThing{{.*}} @thing1{{.*}} @thing2

  struct foo f = {0};
  struct foo g = {0};
  g.big[12] = 12;
  return x & __c11_atomic_compare_exchange_strong(&bigAtomic, &f, g, 5, 5);
  // CHECK: call zeroext i1 @__atomic_compare_exchange(i64 noundef 512, i8* noundef bitcast ({{.*}} @bigAtomic to i8*),
}

// Check that no atomic operations are used in any initialisation of _Atomic
// types.
_Atomic(int) atomic_init_i = 42;

// CHECK: @atomic_init_foo
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

// Check this doesn't crash
// CHECK: @test_atomic_array_param(
void test_atomic_array_param(_Atomic(struct foo) a) {
  test_atomic_array_param(a);
}

#endif
