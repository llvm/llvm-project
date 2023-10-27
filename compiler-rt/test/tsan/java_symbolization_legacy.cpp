// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "java.h"
#include <memory.h>
#include <sanitizer/tsan_interface.h>

#if (__APPLE__)
__attribute__((weak)) // Required for dyld macOS 12.0+
#endif
__attribute__((disable_sanitizer_instrumentation))
extern "C" bool
__tsan_symbolize_external(jptr pc, char *func_buf, jptr func_siz,
                          char *file_buf, jptr file_siz, int *line, int *col) {
  if (pc == (1234 | kExternalPCBit)) {
    memcpy(func_buf, "MyFunc", sizeof("MyFunc"));
    memcpy(file_buf, "MyFile.java", sizeof("MyFile.java"));
    *line = 1234;
    *col = 56;
    return true;
  }
  return false;
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
// CHECK:     #0 MyFunc MyFile.java:1234:56
// CHECK: DONE
