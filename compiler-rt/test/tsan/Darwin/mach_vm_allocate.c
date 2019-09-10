// Test that mach_vm_[de]allocate resets shadow memory status.
//
// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <assert.h>
#include <stdio.h>

#include "../test.h"

static int *global_ptr;
const mach_vm_size_t alloc_size = sizeof(int);

static int *alloc() {
  mach_vm_address_t addr;
  kern_return_t res =
      mach_vm_allocate(mach_task_self(), &addr, alloc_size, VM_FLAGS_ANYWHERE);
  assert(res == KERN_SUCCESS);
  return (int *)addr;
}

static void alloc_fixed(int *ptr) {
  mach_vm_address_t addr = (mach_vm_address_t)ptr;
  kern_return_t res =
      mach_vm_allocate(mach_task_self(), &addr, alloc_size, VM_FLAGS_FIXED);
  assert(res == KERN_SUCCESS);
}

static void dealloc(int *ptr) {
  kern_return_t res =
      mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)ptr, alloc_size);
  assert(res == KERN_SUCCESS);
}

static void *Thread(void *arg) {
  *global_ptr = 7;  // Assignment 1

  // We want to test that TSan does not report a race between the two
  // assignments to global_ptr when memory is re-allocated here. The calls to
  // the API itself are racy though, so ignore them.
  AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  dealloc(global_ptr);
  alloc_fixed(global_ptr);
  AnnotateIgnoreWritesEnd(__FILE__, __LINE__);

  barrier_wait(&barrier);
  return NULL;;
}

int main(int argc, const char *argv[]) {
  barrier_init(&barrier, 2);
  global_ptr = alloc();
  pthread_t t;
  pthread_create(&t, NULL, Thread, NULL);

  barrier_wait(&barrier);
  *global_ptr = 8;  // Assignment 2

  pthread_join(t, NULL);
  dealloc(global_ptr);
  printf("Done.\n");
  return 0;
}

// CHECK: Done.
