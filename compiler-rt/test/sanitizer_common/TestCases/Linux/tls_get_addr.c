// RUN: %clang -g %s -o %t
// RUN: %clang -g %s -DBUILD_SO -fPIC -o %t-so.so -shared
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: glibc

// `__tls_get_addr` is somehow not invoked.
// XFAIL: i386-linux

// These don't intercept __tls_get_addr.
// XFAIL: lsan,hwasan,ubsan

// FIXME: Fails for unknown reasons.
// UNSUPPORTED: powerpc64le-target-arch

#ifndef BUILD_SO
#  include <assert.h>
#  include <dlfcn.h>
#  include <pthread.h>
#  include <stdio.h>
#  include <stdlib.h>

// CHECK-COUNT-2: __sanitizer_get_dtls_size:
size_t __sanitizer_get_dtls_size(const void *ptr)
    __attribute__((disable_sanitizer_instrumentation)) {
  fprintf(stderr, "__sanitizer_get_dtls_size: %p\n", ptr);
  return 0;
}

typedef long *(*get_t)();
get_t GetTls;
void *Thread(void *unused) { return GetTls(); }

int main(int argc, char *argv[]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s-so.so", argv[0]);
  int i;

  void *handle = dlopen(path, RTLD_LAZY);
  if (!handle)
    fprintf(stderr, "%s\n", dlerror());
  assert(handle != 0);
  GetTls = (get_t)dlsym(handle, "GetTls");
  assert(dlerror() == 0);

  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  pthread_join(t, 0);
  pthread_create(&t, 0, Thread, 0);
  pthread_join(t, 0);
  return 0;
}
#else // BUILD_SO
__thread long huge_thread_local_array[1 << 17];
long *GetTls() { return &huge_thread_local_array[0]; }
#endif
