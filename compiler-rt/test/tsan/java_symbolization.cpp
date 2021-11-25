// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "java.h"
#include <memory.h>

extern "C" __attribute__((disable_sanitizer_instrumentation)) void
__tsan_symbolize_external_ex(jptr pc,
                             void (*add_frame)(void *, const char *,
                                               const char *, int, int),
                             void *ctx) {
  if (pc == (1234 | kExternalPCBit)) {
    add_frame(ctx, "MyInnerFunc", "MyInnerFile.java", 1234, 56);
    add_frame(ctx, "MyOuterFunc", "MyOuterFile.java", 4321, 65);
  }
  if (pc == (2345 | kExternalPCBit)) {
    add_frame(ctx, "Caller1", "CallerFile.java", 111, 22);
    add_frame(ctx, "Caller2", "CallerFile.java", 333, 44);
  }
  if (pc == (3456 | kExternalPCBit)) {
    add_frame(ctx, "Allocer1", "Alloc.java", 11, 222);
    add_frame(ctx, "Allocer2", "Alloc.java", 33, 444);
  }
}

void *Thread(void *p) {
  barrier_wait(&barrier);
  __tsan_func_entry(2345 | kExternalPCBit);
  __tsan_write1_pc((jptr)p + 16, 1234 | kExternalPCBit);
  __tsan_func_exit();
  return 0;
}

jptr const kHeapSize = 64 * 1024;
jptr java_heap[kHeapSize];

int main() {
  barrier_init(&barrier, 2);
  jptr jheap = (jptr)java_heap;
  __tsan_java_init(jheap, kHeapSize);
  const int kBlockSize = 32;
  __tsan_func_entry(3456 | kExternalPCBit);
  __tsan_java_alloc(jheap, kBlockSize);
  print_address("addr:", 2, jheap, jheap + 16);
  __tsan_func_exit();
  pthread_t th;
  pthread_create(&th, 0, Thread, (void*)jheap);
  __tsan_func_entry(2345 | kExternalPCBit);
  __tsan_write1_pc(jheap + 16, 1234 | kExternalPCBit);
  __tsan_func_exit();
  barrier_wait(&barrier);
  pthread_join(th, 0);
  __tsan_java_free(jheap, kBlockSize);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK: addr:[[BLOCK:0x[0-9,a-f]+]] [[ADDR:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 1 at [[ADDR]] by thread T1:
// CHECK:     #0 MyInnerFunc MyInnerFile.java:1234:56
// CHECK:     #1 MyOuterFunc MyOuterFile.java:4321:65
// CHECK:     #2 Caller1 CallerFile.java:111:22
// CHECK:     #3 Caller2 CallerFile.java:333:44
// CHECK:   Previous write of size 1 at [[ADDR]] by main thread:
// CHECK:     #0 MyInnerFunc MyInnerFile.java:1234:56
// CHECK:     #1 MyOuterFunc MyOuterFile.java:4321:65
// CHECK:     #2 Caller1 CallerFile.java:111:22
// CHECK:     #3 Caller2 CallerFile.java:333:44
// CHECK:   Location is heap block of size 32 at [[BLOCK]] allocated by main thread:
// CHECK:     #0 Allocer1 Alloc.java:11:222
// CHECK:     #1 Allocer2 Alloc.java:33:444
// CHECK: DONE
