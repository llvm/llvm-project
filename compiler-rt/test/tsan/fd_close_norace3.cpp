// RUN: %clangxx_tsan -O1 %s -o %t && %env_tsan_opts=flush_memory_ms=1 %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void *Thread(void *stop) {
  while (!__atomic_load_n((int *)stop, __ATOMIC_RELAXED))
    close(open("/dev/null", O_RDONLY));
  return 0;
}

int main() {
  int stop = 0;
  const int kThreads = 10;
  pthread_t th[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&th[i], 0, Thread, &stop);
  sleep(5);
  __atomic_store_n(&stop, 1, __ATOMIC_RELAXED);
  for (int i = 0; i < kThreads; i++)
    pthread_join(th[i], 0);
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
