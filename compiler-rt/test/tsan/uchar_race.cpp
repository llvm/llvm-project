// Test TSan interception: detects data races on a shared mbstate_t accessed
// concurrently by `mbrtoc16()` and `c32rtomb()`.
//
// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: darwin, android, netbsd, target={{.*(solaris|aix).*}}

#include <uchar.h>

#include "test.h"

static mbstate_t SharedState;

static void *Thread1(void *x) {
  barrier_wait(&barrier);
  char16_t c16 = 0;
  mbrtoc16(&c16, "a", 1, &SharedState);
  return nullptr;
}

static void *Thread2(void *x) {
  char buf[16];
  c32rtomb(buf, U'b', &SharedState);
  barrier_wait(&barrier);
  return nullptr;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], nullptr, Thread1, nullptr);
  pthread_create(&t[1], nullptr, Thread2, nullptr);
  pthread_join(t[0], nullptr);
  pthread_join(t[1], nullptr);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
