// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s -analyzer-config eagerly-assume=false

// Tests for c11 atomics. Many of these tests currently yield unknown
// because we don't fully model the atomics and instead imprecisely
// treat their arguments as escaping.

typedef unsigned int uint32_t;
typedef enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_consume = __ATOMIC_CONSUME,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

void clang_analyzer_eval(int);

struct RefCountedStruct {
  uint32_t refCount;
  void *ptr;
};

void test_atomic_fetch_add(struct RefCountedStruct *s) {
  s->refCount = 1;

  uint32_t result = __c11_atomic_fetch_add((volatile _Atomic(uint32_t) *)&s->refCount,- 1, memory_order_relaxed);

  // When we model atomics fully this should (probably) be FALSE. It should never
  // be TRUE (because the operation mutates the passed in storage).
  clang_analyzer_eval(s->refCount == 1); // expected-warning {{UNKNOWN}}

  // When fully modeled this should be TRUE
  clang_analyzer_eval(result == 1); // expected-warning {{UNKNOWN}}
}

void test_atomic_load(struct RefCountedStruct *s) {
  s->refCount = 1;

  uint32_t result = __c11_atomic_load((volatile _Atomic(uint32_t) *)&s->refCount, memory_order_relaxed);

  // When we model atomics fully this should (probably) be TRUE.
  clang_analyzer_eval(s->refCount == 1); // expected-warning {{UNKNOWN}}

  // When fully modeled this should be TRUE
  clang_analyzer_eval(result == 1); // expected-warning {{UNKNOWN}}
}

void test_atomic_store(struct RefCountedStruct *s) {
  s->refCount = 1;

  __c11_atomic_store((volatile _Atomic(uint32_t) *)&s->refCount, 2, memory_order_relaxed);

  // When we model atomics fully this should (probably) be FALSE. It should never
  // be TRUE (because the operation mutates the passed in storage).
  clang_analyzer_eval(s->refCount == 1); // expected-warning {{UNKNOWN}}
}

void test_atomic_exchange(struct RefCountedStruct *s) {
  s->refCount = 1;

  uint32_t result = __c11_atomic_exchange((volatile _Atomic(uint32_t) *)&s->refCount, 2, memory_order_relaxed);

  // When we model atomics fully this should (probably) be FALSE. It should never
  // be TRUE (because the operation mutates the passed in storage).
  clang_analyzer_eval(s->refCount == 1); // expected-warning {{UNKNOWN}}

  // When fully modeled this should be TRUE
  clang_analyzer_eval(result == 1); // expected-warning {{UNKNOWN}}
}


void test_atomic_compare_exchange_strong(struct RefCountedStruct *s) {
  s->refCount = 1;
  uint32_t expected = 2;
  uint32_t desired = 3;
  _Bool result = __c11_atomic_compare_exchange_strong((volatile _Atomic(uint32_t) *)&s->refCount, &expected, desired, memory_order_relaxed, memory_order_relaxed);

  // For now we expect both expected and refCount to be invalidated by the
  // call. In the future we should model more precisely.
  clang_analyzer_eval(s->refCount == 3); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(expected == 2); // expected-warning {{UNKNOWN}}
}

void test_atomic_compare_exchange_weak(struct RefCountedStruct *s) {
  s->refCount = 1;
  uint32_t expected = 2;
  uint32_t desired = 3;
  _Bool result = __c11_atomic_compare_exchange_weak((volatile _Atomic(uint32_t) *)&s->refCount, &expected, desired, memory_order_relaxed, memory_order_relaxed);

  // For now we expect both expected and refCount to be invalidated by the
  // call. In the future we should model more precisely.
  clang_analyzer_eval(s->refCount == 3); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(expected == 2); // expected-warning {{UNKNOWN}}
}

// PR49422
void test_atomic_compare(int input) {
  _Atomic(int) x = input;
  if (x > 0) {
    // no crash
  }
}
