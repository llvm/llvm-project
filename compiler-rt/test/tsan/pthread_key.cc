// RUN: %clangxx_tsan -O1 %s -DBUILD_SO -fPIC -shared -o %t-so.so
// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Extracted from:
// https://bugs.chromium.org/p/v8/issues/detail?id=4995

#include "test.h"

void* thr(void* arg) {
  const int N = 32;
  pthread_key_t keys_[N];
  for (size_t i = 0; i < N; ++i) {
    int err = pthread_key_create(&keys_[i], 0);
    if (err) {
      fprintf(stderr, "pthread_key_create failed with %d\n", err);
      exit(1);
    }
  }
  for (size_t i = 0; i < N; i++)
    pthread_setspecific(keys_[i], (void*)(long)i);
  for (size_t i = 0; i < N; i++)
    pthread_key_delete(keys_[i]);
  return 0;
}

int main() {
  for (int i = 0; i < 10; i++) {
    pthread_t th;
    pthread_create(&th, 0, thr, 0);
    pthread_join(th, 0);
  }
  pthread_t th[2];
  pthread_create(&th[0], 0, thr, 0);
  pthread_create(&th[1], 0, thr, 0);
  pthread_join(th[0], 0);
  pthread_join(th[1], 0);
  fprintf(stderr, "DONE\n");
  // CHECK: DONE
}
