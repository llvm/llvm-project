// Test that ASan doesn't deadlock in __interceptor_pthread_create called
// from dlopened shared library constructor. The deadlock happens only in shared
// ASan runtime with recent Glibc (2.23 fits) when __interceptor_pthread_create
// grabs global Glibc's GL(dl_load_lock) and waits for tls_get_addr_tail that
// also tries to acquire it.
//
// RUN: %clangxx_asan -DBUILD_SO=1 -fPIC -shared %s -o %t-so.so
// RUN: %clangxx_asan %s -o %t
// RUN: %run %t 2>&1

// dlopen() can not be intercepted on Android
// UNSUPPORTED: android
// REQUIRES: x86-target-arch

#ifdef BUILD_SO

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void *threadFn(void *) {
  fprintf(stderr, "thread started\n");
  while (true) {
    usleep(100000);
  }
  return 0;
}

void __attribute__((constructor)) startPolling() {
  fprintf(stderr, "initializing library\n");
  pthread_t t;
  pthread_create(&t, 0, &threadFn, 0);
  fprintf(stderr, "done\n");
}

#else

#include <dlfcn.h>
#include <stdlib.h>
#include <string>

int main(int argc, char **argv) {
  std::string path = std::string(argv[0]) + "-so.so";
  void *handle = dlopen(path.c_str(), RTLD_LAZY);
  if (!handle)
    abort();
  return 0;
}
#endif
