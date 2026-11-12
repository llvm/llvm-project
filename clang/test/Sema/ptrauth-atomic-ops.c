// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify -fptrauth-intrinsics %s

#include <stdatomic.h>

int i;
int *__ptrauth(2, 1, 100) authenticated_ptr = &i;
int *__ptrauth(2, 0, 200) non_addr_discriminatedauthenticated_ptr = &i;
int * wat = &i;
long __ptrauth(2, 0, 200) qualified_int;
_Atomic long __ptrauth(2, 0, 200) atomic_qualified_int;
#define ATOMIZE(p) (__typeof__(p) volatile _Atomic *)(long)(&p)

void f() {
  static int j = 1;
  __c11_atomic_init(ATOMIZE(authenticated_ptr), 5);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to a non address discriminated type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __c11_atomic_store(ATOMIZE(authenticated_ptr), 0, memory_order_relaxed);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to a non address discriminated type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __c11_atomic_load(ATOMIZE(authenticated_ptr), memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to a non address discriminated type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __c11_atomic_store(ATOMIZE(authenticated_ptr), 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to a non address discriminated type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_store_n(ATOMIZE(authenticated_ptr), 4, memory_order_release);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to a non address discriminated type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_store(ATOMIZE(authenticated_ptr), j, memory_order_release);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to a non address discriminated type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __c11_atomic_exchange(ATOMIZE(authenticated_ptr), 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to a non address discriminated type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_exchange(ATOMIZE(authenticated_ptr), &j, &j, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to a non address discriminated type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __c11_atomic_fetch_add(ATOMIZE(authenticated_ptr), 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_fetch_add(ATOMIZE(authenticated_ptr), 3, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_fetch_sub(ATOMIZE(authenticated_ptr), 3, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_fetch_min(ATOMIZE(authenticated_ptr), 3, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_fetch_max(ATOMIZE(authenticated_ptr), 3, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __c11_atomic_fetch_and(ATOMIZE(authenticated_ptr), 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_fetch_and(ATOMIZE(authenticated_ptr), 3, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_fetch_or(ATOMIZE(authenticated_ptr), 3, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_fetch_xor(ATOMIZE(authenticated_ptr), 3, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}

  __c11_atomic_init(ATOMIZE(non_addr_discriminatedauthenticated_ptr), &j);
  __c11_atomic_store(ATOMIZE(non_addr_discriminatedauthenticated_ptr), 0, memory_order_relaxed);
  __c11_atomic_load(ATOMIZE(non_addr_discriminatedauthenticated_ptr), memory_order_seq_cst);
  __atomic_store(&j, ATOMIZE(non_addr_discriminatedauthenticated_ptr), memory_order_release);
  // expected-error@-1 {{incompatible pointer types passing 'volatile __ptrauth(2,0,200) _Atomic(int *) *' to parameter of type 'int *'}}
  __c11_atomic_exchange(ATOMIZE(j), ATOMIZE(non_addr_discriminatedauthenticated_ptr), memory_order_seq_cst);
  // expected-error@-1 {{incompatible pointer to integer conversion passing 'volatile __ptrauth(2,0,200) _Atomic(int *) *' to parameter of type 'typeof (j)' (aka 'int')}}
  __c11_atomic_fetch_add(ATOMIZE(non_addr_discriminatedauthenticated_ptr), ATOMIZE(j), memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,0,200) _Atomic(int *) *' invalid)}}
  __c11_atomic_fetch_and(ATOMIZE(j), ATOMIZE(non_addr_discriminatedauthenticated_ptr), memory_order_seq_cst);
  // expected-error@-1 {{incompatible pointer to integer conversion passing 'volatile __ptrauth(2,0,200) _Atomic(int *) *' to parameter of type 'typeof (j)' (aka 'int')}}
  __c11_atomic_fetch_add(ATOMIZE(non_addr_discriminatedauthenticated_ptr), 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,0,200) _Atomic(int *) *' invalid)}}
  __atomic_fetch_add(&non_addr_discriminatedauthenticated_ptr, 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __c11_atomic_fetch_add(&atomic_qualified_int, 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('__ptrauth(2,0,200) _Atomic(long) *' invalid)}}
  __atomic_fetch_add(&qualified_int, 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('__ptrauth(2,0,200) long *' invalid)}}
  __atomic_fetch_or(&atomic_qualified_int, 1, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic arithmetic operation must not be a pointer to a '__ptrauth'-qualified type ('__ptrauth(2,0,200) _Atomic(long) *' invalid)}}

  __atomic_test_and_set(&non_addr_discriminatedauthenticated_ptr, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic test-and-set or clear operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __atomic_clear(&non_addr_discriminatedauthenticated_ptr, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic test-and-set or clear operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __atomic_test_and_set(ATOMIZE(authenticated_ptr), memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic test-and-set or clear operation must not be a pointer to a '__ptrauth'-qualified type ('volatile __ptrauth(2,1,100) _Atomic(int *) *' invalid)}}
  __atomic_clear(&qualified_int, memory_order_seq_cst);
  // expected-error@-1 {{address argument to atomic test-and-set or clear operation must not be a pointer to a '__ptrauth'-qualified type ('__ptrauth(2,0,200) long *' invalid)}}

  (*ATOMIZE(non_addr_discriminatedauthenticated_ptr))++;
  // expected-error@-1 {{atomic increment of '__ptrauth'-qualified type 'volatile __ptrauth(2,0,200) _Atomic(int *)' is not supported}}
  --(*ATOMIZE(non_addr_discriminatedauthenticated_ptr));
  // expected-error@-1 {{atomic decrement of '__ptrauth'-qualified type 'volatile __ptrauth(2,0,200) _Atomic(int *)' is not supported}}
  (*ATOMIZE(non_addr_discriminatedauthenticated_ptr)) += 1;
  // expected-error@-1 {{atomic compound assignment of '__ptrauth'-qualified type 'volatile __ptrauth(2,0,200) _Atomic(int *)' is not supported}}
  (*ATOMIZE(authenticated_ptr))--;
  // expected-error@-1 {{atomic decrement of '__ptrauth'-qualified type 'volatile __ptrauth(2,1,100) _Atomic(int *)' is not supported}}
  atomic_qualified_int++;
  // expected-error@-1 {{atomic increment of '__ptrauth'-qualified type '__ptrauth(2,0,200) _Atomic(long)' is not supported}}
  ++atomic_qualified_int;
  // expected-error@-1 {{atomic increment of '__ptrauth'-qualified type '__ptrauth(2,0,200) _Atomic(long)' is not supported}}
  atomic_qualified_int -= 2;
  // expected-error@-1 {{atomic compound assignment of '__ptrauth'-qualified type '__ptrauth(2,0,200) _Atomic(long)' is not supported}}
  atomic_qualified_int |= 2;
  // expected-error@-1 {{atomic compound assignment of '__ptrauth'-qualified type '__ptrauth(2,0,200) _Atomic(long)' is not supported}}
  qualified_int++;
  qualified_int += 2;
  atomic_qualified_int = 3;


  __sync_fetch_and_add(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_fetch_and_sub(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_fetch_and_or(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_fetch_and_and(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_fetch_and_xor(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_fetch_and_nand(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}

  __sync_add_and_fetch(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_sub_and_fetch(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_or_and_fetch(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_and_and_fetch(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_xor_and_fetch(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_nand_and_fetch(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}

  __sync_bool_compare_and_swap(&authenticated_ptr, 1, 0);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_val_compare_and_swap(&authenticated_ptr, 1, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}

  __sync_lock_test_and_set(&authenticated_ptr, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}
  __sync_lock_release(&authenticated_ptr);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,1,100) *' invalid)}}


int i = 0;

  __sync_fetch_and_add(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_fetch_and_sub(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_fetch_and_or(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_fetch_and_and(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_fetch_and_xor(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}

  __sync_add_and_fetch(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_sub_and_fetch(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_or_and_fetch(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_and_and_fetch(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_xor_and_fetch(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}

  __sync_bool_compare_and_swap(&non_addr_discriminatedauthenticated_ptr, &i, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_val_compare_and_swap(&non_addr_discriminatedauthenticated_ptr, &i, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}

  __sync_lock_test_and_set(&non_addr_discriminatedauthenticated_ptr, &i);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}
  __sync_lock_release(&non_addr_discriminatedauthenticated_ptr);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('int *__ptrauth(2,0,200) *' invalid)}}

  __sync_fetch_and_add(&qualified_int, 1);
  // expected-error@-1 {{address argument to __sync operation must not be a pointer to a '__ptrauth'-qualified type ('__ptrauth(2,0,200) long *' invalid)}}
}

typedef int *__ptrauth(2, 0, 200) UniformPtr;
UniformPtr uniform;
_Atomic UniformPtr atomic_uniform;
_Atomic(int *) atomic_plain;

void by_address(int *p, int **raw, UniformPtr *same) {
  __c11_atomic_compare_exchange_strong(&atomic_uniform, same, p, memory_order_seq_cst, memory_order_seq_cst);
  __atomic_compare_exchange_n(&uniform, same, p, 0, memory_order_seq_cst, memory_order_seq_cst);
  __atomic_compare_exchange(&uniform, same, same, 0, memory_order_seq_cst, memory_order_seq_cst);
  __atomic_load(&uniform, same, memory_order_seq_cst);
  __atomic_store(&uniform, same, memory_order_seq_cst);
  __atomic_exchange(&uniform, same, same, memory_order_seq_cst);
  __typeof__(__atomic_load_n(&uniform, memory_order_seq_cst)) loaded;
  int **unqualified = &loaded;

  __c11_atomic_compare_exchange_strong(&atomic_uniform, raw, p, memory_order_seq_cst, memory_order_seq_cst);
  // expected-error@-1 {{passing 'int **' to parameter of type 'int *__ptrauth(2,0,200) *' changes pointer authentication of pointee type}}
  __atomic_compare_exchange_n(&uniform, raw, p, 0, memory_order_seq_cst, memory_order_seq_cst);
  // expected-error@-1 {{passing 'int **' to parameter of type 'int *__ptrauth(2,0,200) *' changes pointer authentication of pointee type}}
  __atomic_compare_exchange(&uniform, raw, same, 0, memory_order_seq_cst, memory_order_seq_cst);
  // expected-error@-1 {{passing 'int **' to parameter of type 'int *__ptrauth(2,0,200) *' changes pointer authentication of pointee type}}
  __atomic_load(&uniform, raw, memory_order_seq_cst);
  // expected-error@-1 {{passing 'int **' to parameter of type 'int *__ptrauth(2,0,200) *' changes pointer authentication of pointee type}}
  __atomic_exchange(&uniform, raw, same, memory_order_seq_cst);
  // expected-error@-1 {{passing 'int **' to parameter of type 'int *__ptrauth(2,0,200) *' changes pointer authentication of pointee type}}
  __c11_atomic_compare_exchange_strong(&atomic_plain, same, p, memory_order_seq_cst, memory_order_seq_cst);
  // expected-error@-1 {{passing 'UniformPtr *' (aka 'int *__ptrauth(2,0,200) *') to parameter of type 'int **' changes pointer authentication of pointee type}}
}
