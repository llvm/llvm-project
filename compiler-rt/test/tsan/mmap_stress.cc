// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <errno.h>
#include <sys/mman.h>

void *SubWorker(void *arg) {
  (void)arg;
  const int kMmapSize =  65536;
  for (int i = 0; i < 500; i++) {
    int *ptr = (int*)mmap(0, kMmapSize, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (ptr == MAP_FAILED)
      exit(printf("mmap failed: %d\n", errno));
    *ptr = 42;
    if (munmap(ptr, kMmapSize))
      exit(printf("munmap failed: %d\n", errno));
  }
  return 0;
}

void *Worker1(void *arg) {
  (void)arg;
  pthread_t th[4];
  for (int i = 0; i < 4; i++) {
    if (pthread_create(&th[i], 0, SubWorker, 0))
      exit(printf("pthread_create failed: %d\n", errno));
  }
  for (int i = 0; i < 4; i++) {
    if (pthread_join(th[i], 0))
      exit(printf("pthread_join failed: %d\n", errno));
  }
  return 0;
}

void *Worker(void *arg) {
  (void)arg;
  pthread_t th[4];
  for (int i = 0; i < 4; i++) {
    if (pthread_create(&th[i], 0, Worker1, 0))
      exit(printf("pthread_create failed: %d\n", errno));
  }
  for (int i = 0; i < 4; i++) {
    if (pthread_join(th[i], 0))
      exit(printf("pthread_join failed: %d\n", errno));
  }
  return 0;
}

int main() {
  pthread_t th[4];
  for (int i = 0; i < 4; i++) {
    if (pthread_create(&th[i], 0, Worker, 0))
      exit(printf("pthread_create failed: %d\n", errno));
  }
  for (int i = 0; i < 4; i++) {
    if (pthread_join(th[i], 0))
      exit(printf("pthread_join failed: %d\n", errno));
  }
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
