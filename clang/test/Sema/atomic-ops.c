// RUN: %clang_cc1 %s -verify -fsyntax-only -triple=i686-linux-gnu -std=c11

// Basic parsing/Sema tests for __c11_atomic_*

typedef enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
} memory_order;

struct S { char c[3]; };

_Static_assert(__GCC_ATOMIC_BOOL_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_CHAR_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_CHAR16_T_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_CHAR32_T_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_WCHAR_T_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_SHORT_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_INT_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_LONG_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_LLONG_LOCK_FREE == 1, "");
_Static_assert(__GCC_ATOMIC_POINTER_LOCK_FREE == 2, "");

_Static_assert(__c11_atomic_is_lock_free(1), "");
_Static_assert(__c11_atomic_is_lock_free(2), "");
_Static_assert(__c11_atomic_is_lock_free(3), ""); // expected-error {{not an integral constant expression}}
_Static_assert(__c11_atomic_is_lock_free(4), "");
_Static_assert(__c11_atomic_is_lock_free(8), "");
_Static_assert(__c11_atomic_is_lock_free(16), ""); // expected-error {{not an integral constant expression}}
_Static_assert(__c11_atomic_is_lock_free(17), ""); // expected-error {{not an integral constant expression}}

_Static_assert(__atomic_is_lock_free(1, 0), "");
_Static_assert(__atomic_is_lock_free(2, 0), "");
_Static_assert(__atomic_is_lock_free(3, 0), ""); // expected-error {{not an integral constant expression}}
_Static_assert(__atomic_is_lock_free(4, 0), "");
_Static_assert(__atomic_is_lock_free(8, 0), "");
_Static_assert(__atomic_is_lock_free(16, 0), ""); // expected-error {{not an integral constant expression}}
_Static_assert(__atomic_is_lock_free(17, 0), ""); // expected-error {{not an integral constant expression}}

char i8;
short i16;
int i32;
int __attribute__((vector_size(8))) i64;
struct Incomplete *incomplete;

_Static_assert(__atomic_is_lock_free(1, &i8), "");
_Static_assert(__atomic_is_lock_free(1, &i64), "");
_Static_assert(__atomic_is_lock_free(2, &i8), ""); // expected-error {{not an integral constant expression}}
_Static_assert(__atomic_is_lock_free(2, &i16), "");
_Static_assert(__atomic_is_lock_free(2, &i64), "");
_Static_assert(__atomic_is_lock_free(4, &i16), ""); // expected-error {{not an integral constant expression}}
_Static_assert(__atomic_is_lock_free(4, &i32), "");
_Static_assert(__atomic_is_lock_free(4, &i64), "");
_Static_assert(__atomic_is_lock_free(8, &i32), ""); // expected-error {{not an integral constant expression}}
_Static_assert(__atomic_is_lock_free(8, &i64), "");

_Static_assert(__atomic_always_lock_free(1, 0), "");
_Static_assert(__atomic_always_lock_free(2, 0), "");
_Static_assert(!__atomic_always_lock_free(3, 0), "");
_Static_assert(__atomic_always_lock_free(4, 0), "");
_Static_assert(__atomic_always_lock_free(8, 0), "");
_Static_assert(!__atomic_always_lock_free(16, 0), "");
_Static_assert(!__atomic_always_lock_free(17, 0), "");

_Static_assert(__atomic_always_lock_free(1, incomplete), "");
_Static_assert(!__atomic_always_lock_free(2, incomplete), "");
_Static_assert(!__atomic_always_lock_free(4, incomplete), "");

_Static_assert(__atomic_always_lock_free(1, &i8), "");
_Static_assert(__atomic_always_lock_free(1, &i64), "");
_Static_assert(!__atomic_always_lock_free(2, &i8), "");
_Static_assert(__atomic_always_lock_free(2, &i16), "");
_Static_assert(__atomic_always_lock_free(2, &i64), "");
_Static_assert(!__atomic_always_lock_free(4, &i16), "");
_Static_assert(__atomic_always_lock_free(4, &i32), "");
_Static_assert(__atomic_always_lock_free(4, &i64), "");
_Static_assert(!__atomic_always_lock_free(8, &i32), "");
_Static_assert(__atomic_always_lock_free(8, &i64), "");

void f(_Atomic(int) *i, _Atomic(int*) *p, _Atomic(float) *d,
       int *I, int **P, float *D, struct S *s1, struct S *s2) {
  __c11_atomic_init(I, 5); // expected-error {{pointer to _Atomic}}
  __c11_atomic_load(0); // expected-error {{too few arguments to function}}
  __c11_atomic_load(0,0,0); // expected-error {{too many arguments to function}}
  __c11_atomic_store(0,0,0); // expected-error {{first argument to atomic builtin must be a pointer}}
  __c11_atomic_store((int*)0,0,0); // expected-error {{first argument to atomic operation must be a pointer to _Atomic}}

  __c11_atomic_load(i, memory_order_seq_cst);
  __c11_atomic_load(p, memory_order_seq_cst);
  __c11_atomic_load(d, memory_order_seq_cst);

  int load_n_1 = __atomic_load_n(I, memory_order_relaxed);
  int *load_n_2 = __atomic_load_n(P, memory_order_relaxed);
  float load_n_3 = __atomic_load_n(D, memory_order_relaxed); // expected-error {{must be a pointer to integer or pointer}}
  __atomic_load_n(s1, memory_order_relaxed); // expected-error {{must be a pointer to integer or pointer}}

  __atomic_load(i, I, memory_order_relaxed); // expected-error {{must be a pointer to a trivially-copyable type}}
  __atomic_load(I, i, memory_order_relaxed); // expected-warning {{passing '_Atomic(int) *' to parameter of type 'int *'}}
  __atomic_load(I, *P, memory_order_relaxed);
  __atomic_load(I, *P, memory_order_relaxed, 42); // expected-error {{too many arguments}}
  (int)__atomic_load(I, I, memory_order_seq_cst); // expected-error {{operand of type 'void'}}
  __atomic_load(s1, s2, memory_order_acquire);

  __c11_atomic_store(i, 1, memory_order_seq_cst);
  __c11_atomic_store(p, 1, memory_order_seq_cst); // expected-warning {{incompatible integer to pointer conversion}}
  (int)__c11_atomic_store(d, 1, memory_order_seq_cst); // expected-error {{operand of type 'void'}}

  __atomic_store_n(I, 4, memory_order_release);
  __atomic_store_n(I, 4.0, memory_order_release);
  __atomic_store_n(I, P, memory_order_release); // expected-warning {{parameter of type 'int'}}
  __atomic_store_n(i, 1, memory_order_release); // expected-error {{must be a pointer to integer or pointer}}
  __atomic_store_n(s1, *s2, memory_order_release); // expected-error {{must be a pointer to integer or pointer}}

  __atomic_store(I, *P, memory_order_release);
  __atomic_store(s1, s2, memory_order_release);
  __atomic_store(i, I, memory_order_release); // expected-error {{trivially-copyable}}

  int exchange_1 = __c11_atomic_exchange(i, 1, memory_order_seq_cst);
  int exchange_2 = __c11_atomic_exchange(I, 1, memory_order_seq_cst); // expected-error {{must be a pointer to _Atomic}}
  int exchange_3 = __atomic_exchange_n(i, 1, memory_order_seq_cst); // expected-error {{must be a pointer to integer or pointer}}
  int exchange_4 = __atomic_exchange_n(I, 1, memory_order_seq_cst);

  __atomic_exchange(s1, s2, s2, memory_order_seq_cst);
  __atomic_exchange(s1, I, P, memory_order_seq_cst); // expected-warning 2{{parameter of type 'struct S *'}}
  (int)__atomic_exchange(s1, s2, s2, memory_order_seq_cst); // expected-error {{operand of type 'void'}}

  __c11_atomic_fetch_add(i, 1, memory_order_seq_cst);
  __c11_atomic_fetch_add(p, 1, memory_order_seq_cst);
  __c11_atomic_fetch_add(d, 1, memory_order_seq_cst); // expected-error {{must be a pointer to atomic integer or pointer}}

  __atomic_fetch_add(i, 3, memory_order_seq_cst); // expected-error {{pointer to integer or pointer}}
  __atomic_fetch_sub(I, 3, memory_order_seq_cst);
  __atomic_fetch_sub(P, 3, memory_order_seq_cst);
  __atomic_fetch_sub(D, 3, memory_order_seq_cst); // expected-error {{must be a pointer to integer or pointer}}
  __atomic_fetch_sub(s1, 3, memory_order_seq_cst); // expected-error {{must be a pointer to integer or pointer}}

  __c11_atomic_fetch_and(i, 1, memory_order_seq_cst);
  __c11_atomic_fetch_and(p, 1, memory_order_seq_cst); // expected-error {{must be a pointer to atomic integer}}
  __c11_atomic_fetch_and(d, 1, memory_order_seq_cst); // expected-error {{must be a pointer to atomic integer}}

  __atomic_fetch_and(i, 3, memory_order_seq_cst); // expected-error {{pointer to integer}}
  __atomic_fetch_or(I, 3, memory_order_seq_cst);
  __atomic_fetch_xor(P, 3, memory_order_seq_cst); // expected-error {{must be a pointer to integer}}
  __atomic_fetch_or(D, 3, memory_order_seq_cst); // expected-error {{must be a pointer to integer}}
  __atomic_fetch_and(s1, 3, memory_order_seq_cst); // expected-error {{must be a pointer to integer}}

  _Bool cmpexch_1 = __c11_atomic_compare_exchange_strong(i, 0, 1, memory_order_seq_cst, memory_order_seq_cst);
  _Bool cmpexch_2 = __c11_atomic_compare_exchange_strong(p, 0, (int*)1, memory_order_seq_cst, memory_order_seq_cst);
  _Bool cmpexch_3 = __c11_atomic_compare_exchange_strong(d, (int*)0, 1, memory_order_seq_cst, memory_order_seq_cst); // expected-warning {{incompatible pointer types}}

  _Bool cmpexch_4 = __atomic_compare_exchange_n(I, I, 5, 1, memory_order_seq_cst, memory_order_seq_cst);
  _Bool cmpexch_5 = __atomic_compare_exchange_n(I, P, 5, 0, memory_order_seq_cst, memory_order_seq_cst); // expected-warning {{; dereference with *}}
  _Bool cmpexch_6 = __atomic_compare_exchange_n(I, I, P, 0, memory_order_seq_cst, memory_order_seq_cst); // expected-warning {{passing 'int **' to parameter of type 'int'}}

  _Bool cmpexch_7 = __atomic_compare_exchange(I, I, 5, 1, memory_order_seq_cst, memory_order_seq_cst); // expected-warning {{passing 'int' to parameter of type 'int *'}}
  _Bool cmpexch_8 = __atomic_compare_exchange(I, P, I, 0, memory_order_seq_cst, memory_order_seq_cst); // expected-warning {{; dereference with *}}
  _Bool cmpexch_9 = __atomic_compare_exchange(I, I, I, 0, memory_order_seq_cst, memory_order_seq_cst);

  const volatile int flag_k = 0;
  volatile int flag = 0;
  (void)(int)__atomic_test_and_set(&flag_k, memory_order_seq_cst); // expected-warning {{passing 'const volatile int *' to parameter of type 'volatile void *'}}
  (void)(int)__atomic_test_and_set(&flag, memory_order_seq_cst);
  __atomic_clear(&flag_k, memory_order_seq_cst); // expected-warning {{passing 'const volatile int *' to parameter of type 'volatile void *'}}
  __atomic_clear(&flag, memory_order_seq_cst);
  (int)__atomic_clear(&flag, memory_order_seq_cst); // expected-error {{operand of type 'void'}}

  const _Atomic(int) const_atomic;
  __c11_atomic_init(&const_atomic, 0); // expected-error {{first argument to atomic operation must be a pointer to non-const _Atomic type ('const _Atomic(int) *' invalid)}}
  __c11_atomic_store(&const_atomic, 0, memory_order_release); // expected-error {{first argument to atomic operation must be a pointer to non-const _Atomic type ('const _Atomic(int) *' invalid)}}
  __c11_atomic_load(&const_atomic, memory_order_acquire); // expected-error {{first argument to atomic operation must be a pointer to non-const _Atomic type ('const _Atomic(int) *' invalid)}}
}
