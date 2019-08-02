// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "java.h"
#include <memory.h>

extern "C" void __tsan_symbolize_external_ex(
    jptr pc, void (*add_frame)(void *, const char *, const char *, int, int),
    void *ctx) {
  if (pc == (1234 | kExternalPCBit)) {
    add_frame(ctx, "MyInnerFunc", "MyInnerFile.java", 1234, 56);
    add_frame(ctx, "MyOuterFunc", "MyOuterFile.java", 4321, 65);
  }
}

void *Thread(void *p) {
  barrier_wait(&barrier);
  __tsan_write1_pc((jptr)p, 1234 | kExternalPCBit);
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
  __tsan_write1_pc((jptr)jheap, 1234 | kExternalPCBit);
  barrier_wait(&barrier);
  pthread_join(th, 0);
  __tsan_java_free(jheap, kBlockSize);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:     #0 MyInnerFunc MyInnerFile.java:1234:56
// CHECK:     #1 MyOuterFunc MyOuterFile.java:4321:65
// CHECK: DONE
