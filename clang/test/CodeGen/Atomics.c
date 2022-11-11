// Test frontend handling of __sync builtins.
// Modified from a gcc testcase.
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s

signed char sc;
unsigned char uc;
signed short ss;
unsigned short us;
signed int si;
unsigned int ui;
signed long long sll;
unsigned long long ull;
__int128 s128;
unsigned  __int128 u128;

void test_op_ignore (void) // CHECK-LABEL: define{{.*}} void @test_op_ignore
{
  (void) __sync_fetch_and_add (&sc, 1); // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_add (&uc, 1); // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_add (&ss, 1); // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_add (&us, 1); // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_add (&si, 1); // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_add (&ui, 1); // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_add (&sll, 1); // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 8
  (void) __sync_fetch_and_add (&ull, 1); // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 8

  (void) __sync_fetch_and_sub (&sc, 1); // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_sub (&uc, 1); // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_sub (&ss, 1); // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_sub (&us, 1); // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_sub (&si, 1); // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_sub (&ui, 1); // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_sub (&sll, 1); // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 8
  (void) __sync_fetch_and_sub (&ull, 1); // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 8

  (void) __sync_fetch_and_or (&sc, 1); // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_or (&uc, 1); // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_or (&ss, 1); // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_or (&us, 1); // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_or (&si, 1); // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_or (&ui, 1); // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_or (&sll, 1); // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 8
  (void) __sync_fetch_and_or (&ull, 1); // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 8

  (void) __sync_fetch_and_xor (&sc, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_xor (&uc, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_xor (&ss, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_xor (&us, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_xor (&si, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_xor (&ui, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_xor (&sll, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 8
  (void) __sync_fetch_and_xor (&ull, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 8
  (void) __sync_fetch_and_xor (&u128, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 16
  (void) __sync_fetch_and_xor (&s128, 1); // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 16

  (void) __sync_fetch_and_nand (&sc, 1); // CHECK: atomicrmw nand ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_nand (&uc, 1); // CHECK: atomicrmw nand ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_nand (&ss, 1); // CHECK: atomicrmw nand ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_nand (&us, 1); // CHECK: atomicrmw nand ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_nand (&si, 1); // CHECK: atomicrmw nand ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_nand (&ui, 1); // CHECK: atomicrmw nand ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_nand (&sll, 1); // CHECK: atomicrmw nand ptr {{.*}} seq_cst, align 8
  (void) __sync_fetch_and_nand (&ull, 1); // CHECK: atomicrmw nand ptr {{.*}} seq_cst, align 8

  (void) __sync_fetch_and_and (&sc, 1); // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_and (&uc, 1); // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 1
  (void) __sync_fetch_and_and (&ss, 1); // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_and (&us, 1); // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 2
  (void) __sync_fetch_and_and (&si, 1); // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_and (&ui, 1); // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 4
  (void) __sync_fetch_and_and (&sll, 1); // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 8
  (void) __sync_fetch_and_and (&ull, 1); // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 8

}

void test_fetch_and_op (void) // CHECK-LABEL: define{{.*}} void @test_fetch_and_op
{
  sc = __sync_fetch_and_add (&sc, 11); // CHECK: atomicrmw add
  uc = __sync_fetch_and_add (&uc, 11); // CHECK: atomicrmw add
  ss = __sync_fetch_and_add (&ss, 11); // CHECK: atomicrmw add
  us = __sync_fetch_and_add (&us, 11); // CHECK: atomicrmw add
  si = __sync_fetch_and_add (&si, 11); // CHECK: atomicrmw add
  ui = __sync_fetch_and_add (&ui, 11); // CHECK: atomicrmw add
  sll = __sync_fetch_and_add (&sll, 11); // CHECK: atomicrmw add
  ull = __sync_fetch_and_add (&ull, 11); // CHECK: atomicrmw add

  sc = __sync_fetch_and_sub (&sc, 11); // CHECK: atomicrmw sub
  uc = __sync_fetch_and_sub (&uc, 11); // CHECK: atomicrmw sub
  ss = __sync_fetch_and_sub (&ss, 11); // CHECK: atomicrmw sub
  us = __sync_fetch_and_sub (&us, 11); // CHECK: atomicrmw sub
  si = __sync_fetch_and_sub (&si, 11); // CHECK: atomicrmw sub
  ui = __sync_fetch_and_sub (&ui, 11); // CHECK: atomicrmw sub
  sll = __sync_fetch_and_sub (&sll, 11); // CHECK: atomicrmw sub
  ull = __sync_fetch_and_sub (&ull, 11); // CHECK: atomicrmw sub

  sc = __sync_fetch_and_or (&sc, 11); // CHECK: atomicrmw or
  uc = __sync_fetch_and_or (&uc, 11); // CHECK: atomicrmw or
  ss = __sync_fetch_and_or (&ss, 11); // CHECK: atomicrmw or
  us = __sync_fetch_and_or (&us, 11); // CHECK: atomicrmw or
  si = __sync_fetch_and_or (&si, 11); // CHECK: atomicrmw or
  ui = __sync_fetch_and_or (&ui, 11); // CHECK: atomicrmw or
  sll = __sync_fetch_and_or (&sll, 11); // CHECK: atomicrmw or
  ull = __sync_fetch_and_or (&ull, 11); // CHECK: atomicrmw or

  sc = __sync_fetch_and_xor (&sc, 11); // CHECK: atomicrmw xor
  uc = __sync_fetch_and_xor (&uc, 11); // CHECK: atomicrmw xor
  ss = __sync_fetch_and_xor (&ss, 11); // CHECK: atomicrmw xor
  us = __sync_fetch_and_xor (&us, 11); // CHECK: atomicrmw xor
  si = __sync_fetch_and_xor (&si, 11); // CHECK: atomicrmw xor
  ui = __sync_fetch_and_xor (&ui, 11); // CHECK: atomicrmw xor
  sll = __sync_fetch_and_xor (&sll, 11); // CHECK: atomicrmw xor
  ull = __sync_fetch_and_xor (&ull, 11); // CHECK: atomicrmw xor

  sc = __sync_fetch_and_nand (&sc, 11); // CHECK: atomicrmw nand
  uc = __sync_fetch_and_nand (&uc, 11); // CHECK: atomicrmw nand
  ss = __sync_fetch_and_nand (&ss, 11); // CHECK: atomicrmw nand
  us = __sync_fetch_and_nand (&us, 11); // CHECK: atomicrmw nand
  si = __sync_fetch_and_nand (&si, 11); // CHECK: atomicrmw nand
  ui = __sync_fetch_and_nand (&ui, 11); // CHECK: atomicrmw nand
  sll = __sync_fetch_and_nand (&sll, 11); // CHECK: atomicrmw nand
  ull = __sync_fetch_and_nand (&ull, 11); // CHECK: atomicrmw nand

  sc = __sync_fetch_and_and (&sc, 11); // CHECK: atomicrmw and
  uc = __sync_fetch_and_and (&uc, 11); // CHECK: atomicrmw and
  ss = __sync_fetch_and_and (&ss, 11); // CHECK: atomicrmw and
  us = __sync_fetch_and_and (&us, 11); // CHECK: atomicrmw and
  si = __sync_fetch_and_and (&si, 11); // CHECK: atomicrmw and
  ui = __sync_fetch_and_and (&ui, 11); // CHECK: atomicrmw and
  sll = __sync_fetch_and_and (&sll, 11); // CHECK: atomicrmw and
  ull = __sync_fetch_and_and (&ull, 11); // CHECK: atomicrmw and

}

void test_op_and_fetch (void)
{
  sc = __sync_add_and_fetch (&sc, uc); // CHECK: atomicrmw add
  uc = __sync_add_and_fetch (&uc, uc); // CHECK: atomicrmw add
  ss = __sync_add_and_fetch (&ss, uc); // CHECK: atomicrmw add
  us = __sync_add_and_fetch (&us, uc); // CHECK: atomicrmw add
  si = __sync_add_and_fetch (&si, uc); // CHECK: atomicrmw add
  ui = __sync_add_and_fetch (&ui, uc); // CHECK: atomicrmw add
  sll = __sync_add_and_fetch (&sll, uc); // CHECK: atomicrmw add
  ull = __sync_add_and_fetch (&ull, uc); // CHECK: atomicrmw add

  sc = __sync_sub_and_fetch (&sc, uc); // CHECK: atomicrmw sub
  uc = __sync_sub_and_fetch (&uc, uc); // CHECK: atomicrmw sub
  ss = __sync_sub_and_fetch (&ss, uc); // CHECK: atomicrmw sub
  us = __sync_sub_and_fetch (&us, uc); // CHECK: atomicrmw sub
  si = __sync_sub_and_fetch (&si, uc); // CHECK: atomicrmw sub
  ui = __sync_sub_and_fetch (&ui, uc); // CHECK: atomicrmw sub
  sll = __sync_sub_and_fetch (&sll, uc); // CHECK: atomicrmw sub
  ull = __sync_sub_and_fetch (&ull, uc); // CHECK: atomicrmw sub

  sc = __sync_or_and_fetch (&sc, uc); // CHECK: atomicrmw or
  uc = __sync_or_and_fetch (&uc, uc); // CHECK: atomicrmw or
  ss = __sync_or_and_fetch (&ss, uc); // CHECK: atomicrmw or
  us = __sync_or_and_fetch (&us, uc); // CHECK: atomicrmw or
  si = __sync_or_and_fetch (&si, uc); // CHECK: atomicrmw or
  ui = __sync_or_and_fetch (&ui, uc); // CHECK: atomicrmw or
  sll = __sync_or_and_fetch (&sll, uc); // CHECK: atomicrmw or
  ull = __sync_or_and_fetch (&ull, uc); // CHECK: atomicrmw or

  sc = __sync_xor_and_fetch (&sc, uc); // CHECK: atomicrmw xor
  uc = __sync_xor_and_fetch (&uc, uc); // CHECK: atomicrmw xor
  ss = __sync_xor_and_fetch (&ss, uc); // CHECK: atomicrmw xor
  us = __sync_xor_and_fetch (&us, uc); // CHECK: atomicrmw xor
  si = __sync_xor_and_fetch (&si, uc); // CHECK: atomicrmw xor
  ui = __sync_xor_and_fetch (&ui, uc); // CHECK: atomicrmw xor
  sll = __sync_xor_and_fetch (&sll, uc); // CHECK: atomicrmw xor
  ull = __sync_xor_and_fetch (&ull, uc); // CHECK: atomicrmw xor

  sc = __sync_nand_and_fetch (&sc, uc); // CHECK: atomicrmw nand
                                        // CHECK: and
                                        // CHECK: xor
                                        // CHECK: -1
  uc = __sync_nand_and_fetch (&uc, uc); // CHECK: atomicrmw nand
                                        // CHECK: and
                                        // CHECK: xor
                                        // CHECK: -1
  ss = __sync_nand_and_fetch (&ss, uc); // CHECK: atomicrmw nand
                                        // CHECK: and
                                        // CHECK: xor
                                        // CHECK: -1
  us = __sync_nand_and_fetch (&us, uc); // CHECK: atomicrmw nand
                                        // CHECK: and
                                        // CHECK: xor
                                        // CHECK: -1
  si = __sync_nand_and_fetch (&si, uc); // CHECK: atomicrmw nand
                                        // CHECK: and
                                        // CHECK: xor
                                        // CHECK: -1
  ui = __sync_nand_and_fetch (&ui, uc); // CHECK: atomicrmw nand
                                        // CHECK: and
                                        // CHECK: xor
                                        // CHECK: -1
  sll = __sync_nand_and_fetch (&sll, uc); // CHECK: atomicrmw nand
                                          // CHECK: and
                                          // CHECK: xor
                                          // CHECK: -1
  ull = __sync_nand_and_fetch (&ull, uc); // CHECK: atomicrmw nand
                                          // CHECK: and
                                          // CHECK: xor
                                          // CHECK: -1
  u128 = __sync_nand_and_fetch (&u128, uc); // CHECK: atomicrmw nand
                                          // CHECK: and
                                          // CHECK: xor
                                          // CHECK: -1
  s128 = __sync_nand_and_fetch (&s128, uc); // CHECK: atomicrmw nand
                                          // CHECK: and
                                          // CHECK: xor
                                          // CHECK: -1

  sc = __sync_and_and_fetch (&sc, uc); // CHECK: atomicrmw and
  uc = __sync_and_and_fetch (&uc, uc); // CHECK: atomicrmw and
  ss = __sync_and_and_fetch (&ss, uc); // CHECK: atomicrmw and
  us = __sync_and_and_fetch (&us, uc); // CHECK: atomicrmw and
  si = __sync_and_and_fetch (&si, uc); // CHECK: atomicrmw and
  ui = __sync_and_and_fetch (&ui, uc); // CHECK: atomicrmw and
  sll = __sync_and_and_fetch (&sll, uc); // CHECK: atomicrmw and
  ull = __sync_and_and_fetch (&ull, uc); // CHECK: atomicrmw and

}

void test_compare_and_swap (void)
{
  sc = __sync_val_compare_and_swap (&sc, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 1
  // CHECK: extractvalue { i8, i1 } [[PAIR]], 0

  uc = __sync_val_compare_and_swap (&uc, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 1
  // CHECK: extractvalue { i8, i1 } [[PAIR]], 0

  ss = __sync_val_compare_and_swap (&ss, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 2
  // CHECK: extractvalue { i16, i1 } [[PAIR]], 0

  us = __sync_val_compare_and_swap (&us, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 2
  // CHECK: extractvalue { i16, i1 } [[PAIR]], 0

  si = __sync_val_compare_and_swap (&si, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 4
  // CHECK: extractvalue { i32, i1 } [[PAIR]], 0

  ui = __sync_val_compare_and_swap (&ui, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 4
  // CHECK: extractvalue { i32, i1 } [[PAIR]], 0

  sll = __sync_val_compare_and_swap (&sll, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 8
  // CHECK: extractvalue { i64, i1 } [[PAIR]], 0

  ull = __sync_val_compare_and_swap (&ull, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 8
  // CHECK: extractvalue { i64, i1 } [[PAIR]], 0


  ui = __sync_bool_compare_and_swap (&sc, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 1
  // CHECK: extractvalue { i8, i1 } [[PAIR]], 1

  ui = __sync_bool_compare_and_swap (&uc, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 1
  // CHECK: extractvalue { i8, i1 } [[PAIR]], 1

  ui = __sync_bool_compare_and_swap (&ss, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 2
  // CHECK: extractvalue { i16, i1 } [[PAIR]], 1

  ui = __sync_bool_compare_and_swap (&us, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 2
  // CHECK: extractvalue { i16, i1 } [[PAIR]], 1

  ui = __sync_bool_compare_and_swap (&si, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 4
  // CHECK: extractvalue { i32, i1 } [[PAIR]], 1

  ui = __sync_bool_compare_and_swap (&ui, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 4
  // CHECK: extractvalue { i32, i1 } [[PAIR]], 1

  ui = __sync_bool_compare_and_swap (&sll, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 8
  // CHECK: extractvalue { i64, i1 } [[PAIR]], 1

  ui = __sync_bool_compare_and_swap (&ull, uc, sc);
  // CHECK: [[PAIR:%[a-z0-9._]+]] = cmpxchg ptr {{.*}} seq_cst, align 8
  // CHECK: extractvalue { i64, i1 } [[PAIR]], 1
}

void test_lock (void)
{
  sc = __sync_lock_test_and_set (&sc, 1); // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 1
  uc = __sync_lock_test_and_set (&uc, 1); // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 1
  ss = __sync_lock_test_and_set (&ss, 1); // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 2
  us = __sync_lock_test_and_set (&us, 1); // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 2
  si = __sync_lock_test_and_set (&si, 1); // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 4
  ui = __sync_lock_test_and_set (&ui, 1); // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 4
  sll = __sync_lock_test_and_set (&sll, 1); // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 8
  ull = __sync_lock_test_and_set (&ull, 1); // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 8

  __sync_synchronize (); // CHECK: fence seq_cst

  __sync_lock_release (&sc); // CHECK: store atomic {{.*}} release, align 1
  __sync_lock_release (&uc); // CHECK: store atomic {{.*}} release, align 1
  __sync_lock_release (&ss); // CHECK: store atomic {{.*}} release, align 2
  __sync_lock_release (&us); /// CHECK: store atomic {{.*}} release, align 2
  __sync_lock_release (&si); // CHECK: store atomic {{.*}} release, align 4
  __sync_lock_release (&ui); // CHECK: store atomic {{.*}} release, align 4
  __sync_lock_release (&sll); // CHECK: store atomic {{.*}} release, align 8
  __sync_lock_release (&ull); // CHECK: store atomic {{.*}} release, align 8
}

void test_atomic(void) {
  ui = __atomic_fetch_min(&ui, 5, __ATOMIC_RELAXED); // CHECK: atomicrmw umin {{.*}} monotonic, align 4
  si = __atomic_fetch_min(&si, 5, __ATOMIC_SEQ_CST); // CHECK: atomicrmw min {{.*}} seq_cst, align 4
  ui = __atomic_fetch_max(&ui, 5, __ATOMIC_ACQUIRE); // CHECK: atomicrmw umax {{.*}} acquire, align 4
  si = __atomic_fetch_max(&si, 5, __ATOMIC_RELEASE); // CHECK: atomicrmw max {{.*}} release, align 4
}
