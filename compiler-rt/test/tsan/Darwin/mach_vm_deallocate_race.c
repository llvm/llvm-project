// Test that concurrent mach_vm_deallocate / mach_vm_allocate on the same
// virtual address range do not race in the TSan runtime's shadow/meta reset.
//
// Several threads free a region while others reuse the same VA (via an address
// hint) and touch it. If mach_vm_deallocate resets the shadow/meta *after* the
// real deallocation, the freed VA can be handed to a reusing thread mid reset
// -> SEGV at a meta address (or clobbered metadata). Resetting before the real
// deallocation (while we still own the mapping) avoids this.
//
// RUN: %clang_tsan %s -o %t
// RUN: %env_tsan_opts=abort_on_error=0 %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: iossim

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <stdio.h>

static const mach_vm_size_t kSize =
    16 << 20; // >128KB forces ResetRange unmap path
static const int kThreads = 4;
static const long kIters = 1000;
static mach_vm_address_t g_hint; // shared placement hint (the contended VA)

static void *worker(void *arg) {
  for (long i = 0; i < kIters; i++) {
    mach_vm_address_t a = g_hint; // prefer the just-freed VA
    if (mach_vm_allocate(mach_task_self(), &a, kSize, VM_FLAGS_ANYWHERE) !=
        KERN_SUCCESS)
      continue;
    *(volatile char *)a = 1; // touch to check for faults
    mach_vm_deallocate(mach_task_self(), a, kSize);
  }
  return NULL;
}

int main(void) {
  mach_vm_address_t a = 0;
  if (mach_vm_allocate(mach_task_self(), &a, kSize, VM_FLAGS_ANYWHERE) !=
      KERN_SUCCESS)
    return 1;
  g_hint = a;
  mach_vm_deallocate(mach_task_self(), a, kSize);

  pthread_t t[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&t[i], NULL, worker, NULL);
  for (int i = 0; i < kThreads; i++)
    pthread_join(t[i], NULL);

  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK-NOT: ThreadSanitizer
// CHECK: Done.
