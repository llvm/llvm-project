// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct P {
  int x;
  int y;
};

int Helper() {
  try {
    static int i = []() {
      throw P{};
      return 1;
    }();
    return i;
  } catch (P) {
    return 0;
  }
}

void *Thread(void *x) {
  for (int i = 0; i < 1000; ++i) {
    Helper();
  }
  return 0;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread, 0);
  pthread_create(&t[1], 0, Thread, 0);
  pthread_join(t[0], 0);
  pthread_join(t[1], 0);
  fprintf(stderr, "PASS\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
