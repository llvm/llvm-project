// REQUIRES: Clang, profile
// RUN: %clangxx_tsan -fprofile-instr-generate -O3 %s -o %t
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

// Check that we don't report data races on accesses to profile counters that
// have been PHI'd together

#include "test.h"

struct Thing1 {
  void *x;
  Thing1(void *x) : x(x) {}
};

struct Thing2 {
  void *x;
  Thing2(void *x) : x(x) {}
};

static void *thread(void *arg) {
  if (arg)
    return Thing1(arg).x;
  return Thing2(arg).x;
}

int main() {
  pthread_t t[2];
  for (pthread_t &th : t)
    pthread_create(&th, nullptr, thread, nullptr);
  for (pthread_t &th : t)
    pthread_join(th, nullptr);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
