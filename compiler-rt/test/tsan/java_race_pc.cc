// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
// This test fails on powerpc64 big endian.
// The Tsan report is returning wrong information about
// the location of the race.
// XFAIL: powerpc64-unknown-linux-gnu
#include "java.h"

void foobar() {
}

void barbaz() {
}

void *Thread(void *p) {
  barrier_wait(&barrier);
  __tsan_read1_pc((jptr)p, (jptr)foobar + kPCInc);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  int const kHeapSize = 1024 * 1024;
  jptr jheap = (jptr)malloc(kHeapSize + 8) + 8;
  __tsan_java_init(jheap, kHeapSize);
  const int kBlockSize = 16;
  __tsan_java_alloc(jheap, kBlockSize);
  pthread_t th;
  pthread_create(&th, 0, Thread, (void*)jheap);
  __tsan_write1_pc((jptr)jheap, (jptr)barbaz + kPCInc);
  barrier_wait(&barrier);
  pthread_join(th, 0);
  __tsan_java_free(jheap, kBlockSize);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:     #0 foobar
// CHECK:     #0 barbaz
// CHECK: DONE
