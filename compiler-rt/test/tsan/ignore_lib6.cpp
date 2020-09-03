// RUN: rm -rf %t-dir
// RUN: mkdir %t-dir
// RUN: %clangxx_tsan -O1 %s -DLIB -fPIC -fno-sanitize=thread -shared -o %t-dir/libignore_lib.so
// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t-dir/executable
// RUN: %env_tsan_opts=suppressions='%s.supp' %run %t-dir/executable 2>&1 | FileCheck %s

// Copied from ignore_lib5.cpp:
// REQUIRES: stable-runtime
// UNSUPPORTED: powerpc64le
// UNSUPPORTED: netbsd

// Test that pthread_detach works in libraries ignored by called_from_lib.
// For more context see:
// https://groups.google.com/forum/#!topic/thread-sanitizer/ecH2P0QUqPs

#include "test.h"
#include <dlfcn.h>
#include <errno.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>

#ifndef LIB

void *thr(void *arg) {
  *(volatile long long *)arg = 1;
  return 0;
}

int main(int argc, char **argv) {
  std::string lib = std::string(dirname(argv[0])) + "/libignore_lib.so";
  void *h = dlopen(lib.c_str(), RTLD_GLOBAL | RTLD_NOW);
  if (h == 0)
    exit(printf("failed to load the library (%d)\n", errno));
  void (*libfunc)() = (void (*)())dlsym(h, "libfunc");
  if (libfunc == 0)
    exit(printf("failed to find the func (%d)\n", errno));
  libfunc();

  const int kThreads = 10;
  pthread_t t[kThreads];
  volatile long long data[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&t[i], 0, thr, (void *)&data[i]);
  for (int i = 0; i < kThreads; i++) {
    pthread_join(t[i], 0);
    data[i] = 2;
  }
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer:

#else // #ifdef LIB

void *thr(void *p) {
  sleep(1);
  pthread_detach(pthread_self());
  return 0;
}

extern "C" void libfunc() {
  const int kThreads = 10;
  pthread_t t[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&t[i], 0, thr, 0);
  sleep(2);
}

#endif // #ifdef LIB
