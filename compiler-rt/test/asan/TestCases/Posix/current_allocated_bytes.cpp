// RUN: %clangxx_asan -O0 %s -pthread -o %t && %run %t
// RUN: %clangxx_asan -O2 %s -pthread -o %t && %run %t
// REQUIRES: stable-runtime

#include <assert.h>
#include <pthread.h>
#include <sanitizer/allocator_interface.h>
#include <stdio.h>
#include <stdlib.h>

const size_t kLargeAlloc = 1UL << 20;

void* allocate(void *arg) {
  volatile void *ptr = malloc(kLargeAlloc);
  free((void*)ptr);
  return 0;
}

void* check_stats(void *arg) {
  assert(__sanitizer_get_current_allocated_bytes() >= 0);
  return 0;
}

int main() {
  size_t used_mem = __sanitizer_get_current_allocated_bytes();
  printf("Before: %zu\n", used_mem);
  const int kNumIterations = 1000;
  for (int iter = 0; iter < kNumIterations; iter++) {
    pthread_t thr[4];
    for (int j = 0; j < 4; j++) {
      assert(0 ==
             pthread_create(&thr[j], 0, (j < 2) ? allocate : check_stats, 0));
    }
    for (int j = 0; j < 4; j++)
      assert(0 == pthread_join(thr[j], 0));
    used_mem = __sanitizer_get_current_allocated_bytes();
    if (used_mem > kLargeAlloc) {
      printf("After iteration %d: %zu\n", iter, used_mem);
      return 1;
    }
  }
  printf("Success after %d iterations\n", kNumIterations);
  return 0;
}
