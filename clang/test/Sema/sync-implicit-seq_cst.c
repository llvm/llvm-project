// RUN: %clang_cc1 %s -verify -ffreestanding -fsyntax-only -triple=i686-linux-gnu -std=c11 -Watomic-implicit-seq-cst -Wno-sync-fetch-and-nand-semantics-changed

// __sync_* operations are implicitly sequentially-consistent. Some codebases
// want to force explicit usage of memory order instead.

void fetch_and_add(int *ptr, int val) { __sync_fetch_and_add(ptr, val); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void fetch_and_sub(int *ptr, int val) { __sync_fetch_and_sub(ptr, val); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void fetch_and_or(int *ptr, int val) { __sync_fetch_and_or(ptr, val); }     // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void fetch_and_and(int *ptr, int val) { __sync_fetch_and_and(ptr, val); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void fetch_and_xor(int *ptr, int val) { __sync_fetch_and_xor(ptr, val); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void fetch_and_nand(int *ptr, int val) { __sync_fetch_and_nand(ptr, val); } // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}

void add_and_fetch(int *ptr, int val) { __sync_add_and_fetch(ptr, val); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void sub_and_fetch(int *ptr, int val) { __sync_sub_and_fetch(ptr, val); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void or_and_fetch(int *ptr, int val) { __sync_or_and_fetch(ptr, val); }     // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void and_and_fetch(int *ptr, int val) { __sync_and_and_fetch(ptr, val); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void xor_and_fetch(int *ptr, int val) { __sync_xor_and_fetch(ptr, val); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void nand_and_fetch(int *ptr, int val) { __sync_nand_and_fetch(ptr, val); } // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}

void bool_compare_and_swap(int *ptr, int oldval, int newval) { __sync_bool_compare_and_swap(ptr, oldval, newval); } // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void val_compare_and_swap(int *ptr, int oldval, int newval) { __sync_val_compare_and_swap(ptr, oldval, newval); }   // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}

void synchronize(void) { __sync_synchronize(); } // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}

void lock_test_and_set(int *ptr, int val) { __sync_lock_test_and_set(ptr, val); } // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
void lock_release(int *ptr) { __sync_lock_release(ptr); }                         // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
