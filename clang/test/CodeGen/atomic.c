// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | FileCheck %s

// CHECK: @[[NONSTATIC_GLOB_POINTER_FROM_INT:.+]] = global ptr null
// CHECK: @[[GLOB_POINTER:.+]] = internal global ptr null
// CHECK: @[[GLOB_POINTER_FROM_INT:.+]] = internal global ptr null
// CHECK: @[[GLOB_INT:.+]] = internal global i32 0
// CHECK: @[[GLOB_FLT:.+]] = internal global float {{[0e\+-\.]+}}, align

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
  // CHECK: atomicrmw add ptr %val, i32 1 seq_cst, align 4
  
  old = __sync_fetch_and_sub(&valc, 2);
  // CHECK: atomicrmw sub ptr %valc, i8 2 seq_cst, align 1
  
  old = __sync_fetch_and_min(&val, 3);
  // CHECK: atomicrmw min ptr %val, i32 3 seq_cst, align 4
  
  old = __sync_fetch_and_max(&val, 4);
  // CHECK: atomicrmw max ptr %val, i32 4 seq_cst, align 4
  
  old = __sync_fetch_and_umin(&uval, 5u);
  // CHECK: atomicrmw umin ptr %uval, i32 5 seq_cst, align 4
  
  old = __sync_fetch_and_umax(&uval, 6u);
  // CHECK: atomicrmw umax ptr %uval, i32 6 seq_cst, align 4
  
  old = __sync_lock_test_and_set(&val, 7);
  // CHECK: atomicrmw xchg ptr %val, i32 7 seq_cst, align 4
  
  old = __sync_swap(&val, 8);
  // CHECK: atomicrmw xchg ptr %val, i32 8 seq_cst, align 4
  
  old = __sync_val_compare_and_swap(&val, 4, 1976);
  // CHECK: [[PAIR:%[a-z0-9_.]+]] = cmpxchg ptr %val, i32 4, i32 1976 seq_cst seq_cst, align 4
  // CHECK: extractvalue { i32, i1 } [[PAIR]], 0

  old = __sync_bool_compare_and_swap(&val, 4, 1976);
  // CHECK: [[PAIR:%[a-z0-9_.]+]] = cmpxchg ptr %val, i32 4, i32 1976 seq_cst seq_cst, align 4
  // CHECK: extractvalue { i32, i1 } [[PAIR]], 1

  old = __sync_fetch_and_and(&val, 0x9);
  // CHECK: atomicrmw and ptr %val, i32 9 seq_cst, align 4

  old = __sync_fetch_and_or(&val, 0xa);
  // CHECK: atomicrmw or ptr %val, i32 10 seq_cst, align 4

  old = __sync_fetch_and_xor(&val, 0xb);
  // CHECK: atomicrmw xor ptr %val, i32 11 seq_cst, align 4
 
  old = __sync_fetch_and_nand(&val, 0xc);
  // CHECK: atomicrmw nand ptr %val, i32 12 seq_cst, align 4
 
  old = __sync_add_and_fetch(&val, 1);
  // CHECK: atomicrmw add ptr %val, i32 1 seq_cst, align 4

  old = __sync_sub_and_fetch(&val, 2);
  // CHECK: atomicrmw sub ptr %val, i32 2 seq_cst, align 4

  old = __sync_and_and_fetch(&valc, 3);
  // CHECK: atomicrmw and ptr %valc, i8 3 seq_cst, align 1

  old = __sync_or_and_fetch(&valc, 4);
  // CHECK: atomicrmw or ptr %valc, i8 4 seq_cst, align 1

  old = __sync_xor_and_fetch(&valc, 5);
  // CHECK: atomicrmw xor ptr %valc, i8 5 seq_cst, align 1
 
  old = __sync_nand_and_fetch(&valc, 6);
  // CHECK: atomicrmw nand ptr %valc, i8 6 seq_cst, align 1
 
  __sync_val_compare_and_swap((void **)0, (void *)0, (void *)0);
  // CHECK: [[PAIR:%[a-z0-9_.]+]] = cmpxchg ptr null, i32 0, i32 0 seq_cst seq_cst, align 4
  // CHECK: extractvalue { i32, i1 } [[PAIR]], 0

  if ( __sync_val_compare_and_swap(&valb, 0, 1)) {
    // CHECK: [[PAIR:%[a-z0-9_.]+]] = cmpxchg ptr %valb, i8 0, i8 1 seq_cst seq_cst, align 1
    // CHECK: [[VAL:%[a-z0-9_.]+]] = extractvalue { i8, i1 } [[PAIR]], 0
    // CHECK: trunc i8 [[VAL]] to i1
    old = 42;
  }
  
  __sync_bool_compare_and_swap((void **)0, (void *)0, (void *)0);
  // CHECK: cmpxchg ptr null, i32 0, i32 0 seq_cst seq_cst, align 4
  
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
  // CHECK: cmpxchg ptr addrspace(256){{.*}}, i32 0, i32 1 seq_cst seq_cst, align 4

  __sync_val_compare_and_swap(P, 0, 1);
  // CHECK: cmpxchg ptr addrspace(256){{.*}}, i32 0, i32 1 seq_cst seq_cst, align 4

  __sync_xor_and_fetch(P, 123);
  // CHECK: atomicrmw xor ptr addrspace(256){{.*}}, i32 123 seq_cst, align 4
}

// Ensure that global initialization of atomics is correct.
static _Atomic(int *) glob_pointer = (void *)0;
static _Atomic(int *) glob_pointer_from_int = 0;
_Atomic(int *) nonstatic_glob_pointer_from_int = 0LL;
static _Atomic int glob_int = 0;
static _Atomic float glob_flt = 0.0f;

void force_global_uses(void) {
  (void)glob_pointer;
  // CHECK: %[[LOCAL_INT:.+]] = load atomic i32, ptr @[[GLOB_POINTER]] seq_cst
  // CHECK-NEXT: inttoptr i32 %[[LOCAL_INT]] to ptr
  (void)glob_pointer_from_int;
  // CHECK: %[[LOCAL_INT_2:.+]] = load atomic i32, ptr @[[GLOB_POINTER_FROM_INT]] seq_cst
  // CHECK-NEXT: inttoptr i32 %[[LOCAL_INT_2]] to ptr
  (void)nonstatic_glob_pointer_from_int;
  // CHECK: %[[LOCAL_INT_3:.+]] = load atomic i32, ptr @[[NONSTATIC_GLOB_POINTER_FROM_INT]] seq_cst
  // CHECK-NEXT: inttoptr i32 %[[LOCAL_INT_3]] to ptr
  (void)glob_int;
  // CHECK: load atomic i32, ptr @[[GLOB_INT]] seq_cst
  (void)glob_flt;
  // CHECK: %[[LOCAL_FLT:.+]] = load atomic i32, ptr @[[GLOB_FLT]] seq_cst
  // CHECK-NEXT: bitcast i32 %[[LOCAL_FLT]] to float
}
