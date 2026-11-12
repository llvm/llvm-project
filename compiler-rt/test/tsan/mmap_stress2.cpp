// RUN: %clang_tsan -O1 %s -o %t && %env_tsan_opts=report_atomic_races=0 %run %t 2>&1 | FileCheck %s
// This test exposed non-atomicity in mmap interceptor
// which made shadow for the region temporarily unmapped.
// This resulted in crashes in the Accesser thread.
#include "test.h"
#include <errno.h>
#include <sys/mman.h>

// The size needs to be large enough to trigger
// large region optimization in the runtime.
const size_t kMmapSize = 16 << 20;

void *Remapper(void *arg) {
  for (;;) {
    void *p = mmap(arg, kMmapSize, PROT_READ | PROT_WRITE,
                   MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED)
      exit(printf("mmap failed: %d\n", errno));
  }
  return 0;
}

void *Accesser(void *arg) {
  unsigned rnd = time(0);
  for (;;) {
    int index = rand_r(&rnd) % kMmapSize;
    char *p = &((char *)arg)[index];
    __atomic_fetch_add(p, 1, __ATOMIC_ACQ_REL);
  }
  return 0;
}

int main() {
  void *p = mmap(0, kMmapSize, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANON, -1, 0);
  if (p == MAP_FAILED)
    exit(printf("mmap failed: %d\n", errno));
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  pthread_t th[2];
  if (pthread_create(&th[0], &attr, Remapper, p))
    exit(printf("pthread_create failed: %d\n", errno));
  if (pthread_create(&th[1], &attr, Accesser, p))
    exit(printf("pthread_create failed: %d\n", errno));
  pthread_attr_destroy(&attr);
  sleep(3);
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
