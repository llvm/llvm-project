// Test that we don't crash accessing DTLS from malloc hook.

// RUN: %clang %s -o %t
// RUN: %clang %s -DBUILD_SO -fPIC -o %t-so.so -shared
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: glibc

// No allocator and hooks.
// XFAIL: ubsan

// FIXME: Crashes on CHECK.
// XFAIL: asan && !i386-linux
// XFAIL: msan && !i386-linux

#ifndef BUILD_SO
#  include <assert.h>
#  include <dlfcn.h>
#  include <pthread.h>
#  include <stdio.h>
#  include <stdlib.h>

typedef long *(*get_t)();
get_t GetTls;
void *Thread(void *unused) { return GetTls(); }

__thread long recursive_hook;

// CHECK: __sanitizer_malloc_hook:
void __sanitizer_malloc_hook(const volatile void *ptr, size_t sz)
    __attribute__((disable_sanitizer_instrumentation)) {
  ++recursive_hook;
  if (recursive_hook == 1 && GetTls)
    fprintf(stderr, "__sanitizer_malloc_hook: %p\n", GetTls());
  --recursive_hook;
}

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
