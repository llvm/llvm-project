// RUN: %clangxx_tsan -O0 %s -o %t
// RUN: not %run %t        2>&1 | FileCheck %s --check-prefix=CHECK-RACE
// RUN:     %run %t ignore 2>&1 | FileCheck %s --check-prefix=CHECK-IGNORE
// XFAIL: netbsd

#include <sys/mman.h>
#include <string.h>
#include <assert.h>
#include <atomic>

#include "test.h"

// Use atomic to ensure we do not have a race for the pointer value itself.  We
// only want to check races in the mmap'd memory to isolate the test that mmap
// respects ignore annotations.
std::atomic<int*> global_p;

void mmap_ignored(bool ignore) {
  const size_t kSize = sysconf(_SC_PAGESIZE);

  if (ignore) AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  void *p = mmap(0, kSize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
  if (ignore) AnnotateIgnoreWritesEnd(__FILE__, __LINE__);

  // Use relaxed to retain the race between the mmap call and the memory write
  global_p.store((int *)p, std::memory_order_relaxed);
  barrier_wait(&barrier);
}

void *WriteToMemory(void *unused) {
  barrier_wait(&barrier);
  global_p[0] = 7;
  return 0;
}

// Create race between allocating (mmap) and writing memory
int main(int argc, const char *argv[]) {
  bool ignore = (argc > 1) && (strcmp(argv[1], "ignore") == 0);

  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, WriteToMemory, 0);
  mmap_ignored(ignore);
  pthread_join(t, 0);

  assert(global_p[0] == 7);
  printf("OK\n");
  return 0;
}

// CHECK-RACE: WARNING: ThreadSanitizer: data race
// CHECK-RACE: OK
// CHECK-IGNORE-NOT: WARNING: ThreadSanitizer: data race
// CHECK-IGNORE: OK
