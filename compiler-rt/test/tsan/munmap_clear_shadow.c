// RUN: %clang_tsan %s -o %t && %run %t | FileCheck %s

// In these systems, the behavior of ReleaseMemoryPagesToOS is madvise(beg, end, MADV_FREE),
// which tags the relevant pages as 'FREE' and does not release them immediately.
// Therefore, we cannot assume that __tsan_read1 will not race with the shadow cleared.
// UNSUPPORTED: darwin,target={{.*(freebsd|netbsd|solaris|haiku).*}}

#include "test.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

void __tsan_read1(void *addr);

struct thread_params {
  char *buf;
  unsigned int size;
};

static void *thread_func(void *arg) {
  struct thread_params *p = (struct thread_params *)arg;
  // Access 1
  p->buf[0] = 0x42;
  p->buf[p->size - 1] = 0x42;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  const unsigned int kPageSize = sysconf(_SC_PAGESIZE);
  // The relevant shadow memory size should be exactly multiple of kPageSize,
  // even if Size = kPageSize - 1.
  const unsigned int Size = kPageSize - 1;

  barrier_init(&barrier, 2);
  char *buf = (char *)mmap(NULL, Size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(buf != MAP_FAILED);
  assert(((uintptr_t)buf % kPageSize) == 0);

  pthread_t t;
  struct thread_params p = {buf, Size};
  pthread_create(&t, 0, thread_func, &p);

  barrier_wait(&barrier);
  // Should clear all the shadow memory related to the mmaped memory.
  munmap(buf, Size);

  // If the shadow memory is cleared completely, the following reads should not
  // cause races and behave the same. However, previously, __tsan_read1(&buf[0])
  // would not report a race, while __tsan_read1(&buf[Size - 1]) did.
  // CHECK-NOT: WARNING: ThreadSanitizer: data race
  __tsan_read1(&buf[0]);        // Access 2
  __tsan_read1(&buf[Size - 1]); // Access 2
  pthread_join(t, 0);

  puts("DONE");

  return 0;
}
