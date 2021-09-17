// Test that dyld interposition works in the presence of DYLD_LIBRARY_PATH.

// RUN: %clang_tsan %s -o %t
// RUN: env DYLD_LIBRARY_PATH=/usr/lib/system/introspection/ %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <pthread.h>
#include <stdio.h>

void *Thread(void *a) {
  fprintf(stderr, "Hello from pthread\n");
  return NULL;
}

int main() {
  pthread_t t;
  pthread_create(&t, NULL, Thread, NULL);
  pthread_join(t, NULL);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello from pthread
// CHECK: Done.
