// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Use of gethostent provokes caching of some resources inside of libc.
// They are freed in __libc_thread_freeres very late in thread lifetime,
// after our ThreadFinish. __libc_thread_freeres calls free which
// previously crashed in malloc hooks.

#include "../test.h"
#include <netdb.h>

long X;

extern "C" void __sanitizer_malloc_hook(void *ptr, size_t size) {
  __atomic_fetch_add(&X, 1, __ATOMIC_RELAXED);
}

extern "C" void __sanitizer_free_hook(void *ptr) {
  __atomic_fetch_sub(&X, 1, __ATOMIC_RELAXED);
}

void *Thread(void *x) {
  sethostent(1);
  gethostbyname("llvm.org");
  gethostent();
  endhostent();
  return NULL;
}

int main() {
  pthread_t th;
  pthread_create(&th, NULL, Thread, NULL);
  pthread_join(th, NULL);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
