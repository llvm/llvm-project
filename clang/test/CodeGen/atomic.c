// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | FileCheck %s

int atomic(void) {
  // non-sensical test for sync functions
  int old;
  int val = 1;
  char valc = 1;
  _Bool valb = 0;
  unsigned int uval = 1;
  int cmp = 0;
  int* ptrval;

  old = __sync_fetch_and_add(&val, 1);
  // CHECK: atomicrmw add i32* %val, i32 1 seq_cst
  
  old = __sync_fetch_and_sub(&valc, 2);
  // CHECK: atomicrmw sub i8* %valc, i8 2 seq_cst
  
  old = __sync_fetch_and_min(&val, 3);
  // CHECK: atomicrmw min i32* %val, i32 3 seq_cst
  
  old = __sync_fetch_and_max(&val, 4);
  // CHECK: atomicrmw max i32* %val, i32 4 seq_cst
  
  old = __sync_fetch_and_umin(&uval, 5u);
  // CHECK: atomicrmw umin i32* %uval, i32 5 seq_cst
  
  old = __sync_fetch_and_umax(&uval, 6u);
  // CHECK: atomicrmw umax i32* %uval, i32 6 seq_cst
  
  old = __sync_lock_test_and_set(&val, 7);
  // CHECK: atomicrmw xchg i32* %val, i32 7 seq_cst
  
  old = __sync_swap(&val, 8);
  // CHECK: atomicrmw xchg i32* %val, i32 8 seq_cst
  
  old = __sync_val_compare_and_swap(&val, 4, 1976);
  // CHECK: cmpxchg i32* %val, i32 4, i32 1976 seq_cst
  
  old = __sync_bool_compare_and_swap(&val, 4, 1976);
  // CHECK: cmpxchg i32* %val, i32 4, i32 1976 seq_cst

  old = __sync_fetch_and_and(&val, 0x9);
  // CHECK: atomicrmw and i32* %val, i32 9 seq_cst

  old = __sync_fetch_and_or(&val, 0xa);
  // CHECK: atomicrmw or i32* %val, i32 10 seq_cst

  old = __sync_fetch_and_xor(&val, 0xb);
  // CHECK: atomicrmw xor i32* %val, i32 11 seq_cst
  
  old = __sync_add_and_fetch(&val, 1);
  // CHECK: atomicrmw add i32* %val, i32 1 seq_cst

  old = __sync_sub_and_fetch(&val, 2);
  // CHECK: atomicrmw sub i32* %val, i32 2 seq_cst

  old = __sync_and_and_fetch(&valc, 3);
  // CHECK: atomicrmw and i8* %valc, i8 3 seq_cst

  old = __sync_or_and_fetch(&valc, 4);
  // CHECK: atomicrmw or i8* %valc, i8 4 seq_cst

  old = __sync_xor_and_fetch(&valc, 5);
  // CHECK: atomicrmw xor i8* %valc, i8 5 seq_cst  
  
  __sync_val_compare_and_swap((void **)0, (void *)0, (void *)0);
  // CHECK: cmpxchg i32* null, i32 0, i32 0 seq_cst

  if ( __sync_val_compare_and_swap(&valb, 0, 1)) {
    // CHECK: cmpxchg i8* %valb, i8 0, i8 1 seq_cst
    old = 42;
  }
  
  __sync_bool_compare_and_swap((void **)0, (void *)0, (void *)0);
  // CHECK: cmpxchg i32* null, i32 0, i32 0 seq_cst
  
  __sync_lock_release(&val);
  // CHECK: store atomic i32 0, {{.*}} release, align 4

  __sync_lock_release(&ptrval);
  // CHECK: store atomic i32 0, {{.*}} release, align 4

  __sync_synchronize ();
  // CHECK: fence seq_cst

  return old;
}

// CHECK: @release_return
void release_return(int *lock) {
  // Ensure this is actually returning void all the way through.
  return __sync_lock_release(lock);
  // CHECK: store atomic {{.*}} release, align 4
}


// rdar://8461279 - Atomics with address spaces.
// CHECK: @addrspace
void addrspace(int  __attribute__((address_space(256))) * P) {
  __sync_bool_compare_and_swap(P, 0, 1);
  // CHECK: cmpxchg i32 addrspace(256)*{{.*}}, i32 0, i32 1 seq_cst  

  __sync_val_compare_and_swap(P, 0, 1);
  // CHECK: cmpxchg i32 addrspace(256)*{{.*}}, i32 0, i32 1 seq_cst  

  __sync_xor_and_fetch(P, 123);
  // CHECK: atomicrmw xor i32 addrspace(256)*{{.*}}, i32 123 seq_cst  
}
