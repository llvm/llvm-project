// RUN: %clang_tsan %s -shared -DBUILD_LIB -o %t.dylib
// RUN: %clang_tsan %s -o %t
// RUN: %run %t %t.dylib 2>&1 | FileCheck %s

// The dylib path is a host path.
// UNSUPPORTED: ios && !iossim

#include <assert.h>
#include <stdlib.h>

#ifdef BUILD_LIB

static int *cleanup_count;

static void cleanup(void) { ++*cleanup_count; }

int register_cleanup(int *count) {
  cleanup_count = count;
  return atexit(cleanup);
}

#else

#  include <dlfcn.h>
#  include <stdio.h>

static int cleanup_count;

static void process_cleanup(void) {
  assert(cleanup_count == 1);
  fprintf(stderr, "process cleanup\n");
}

int main(int argc, char **argv) {
  assert(argc == 2);
  void *image = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  assert(image);
  int (*register_cleanup)(int *) =
      (int (*)(int *))dlsym(image, "register_cleanup");
  assert(register_cleanup);
  int res = register_cleanup(&cleanup_count);
  assert(res == 0);
  res = atexit(process_cleanup);
  assert(res == 0);
  res = dlclose(image);
  assert(res == 0);
  assert(cleanup_count == 1);
  fprintf(stderr, "library cleanup before dlclose returned\n");
  return 0;
}

#endif

// CHECK: library cleanup before dlclose returned
// CHECK-NEXT: process cleanup
