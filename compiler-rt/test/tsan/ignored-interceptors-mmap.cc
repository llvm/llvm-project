// RUN: %clangxx_tsan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NORMAL
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-IGNORE

#include <errno.h>
#include <sys/mman.h>

#include "test.h"

extern "C" {
void AnnotateIgnoreReadsBegin(const char *f, int l);
void AnnotateIgnoreReadsEnd(const char *f, int l);
void AnnotateIgnoreWritesBegin(const char *f, int l);
void AnnotateIgnoreWritesEnd(const char *f, int l);
}

void *global_p;

int mmap_and_ignore_reads_and_writes() {
  const size_t kSize = sysconf(_SC_PAGESIZE);
  void *p = mmap(0, kSize, PROT_READ|PROT_WRITE,
                 MAP_PRIVATE|MAP_ANON, -1, 0);
  if (p == MAP_FAILED)
    return printf("mmap failed with %d\n", errno);
  munmap(p, kSize);

  void *new_p = mmap(p, kSize, PROT_READ|PROT_WRITE,
                     MAP_PRIVATE|MAP_ANON, -1, 0);
  if (p == MAP_FAILED || p != new_p)
    return printf("second mmap failed with %d\n", errno);

  AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  global_p = p;
  AnnotateIgnoreWritesEnd(__FILE__, __LINE__);
  barrier_wait(&barrier);
  return 0;
}

void *Thread(void *a) {
  barrier_wait(&barrier);

  ((int*)global_p)[1] = 10;
  printf("Read the zero value from mmapped memory %d\n", ((int*)global_p)[1]);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  if (mmap_and_ignore_reads_and_writes())
    return 1;
  pthread_join(t, 0);
  printf("OK\n");
  return 0;
}

// CHECK-NORMAL: WARNING: ThreadSanitizer: data race
// CHECK-NORMAL: OK
// CHECK-IGNORE_NOT: WARNING: ThreadSanitizer: data race
// CHECK-IGNORE: OK
