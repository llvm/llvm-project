// RUN: %clangxx -g -O0 %s -o %t

// Check that trying to dlopen() the ASan dylib fails.
// We explictly set `abort_on_error=0` because
// - By default the lit config sets this but we don't want this
//   test to implicitly depend on this.
// - It avoids requiring `--crash` to be passed to `not`.
// RUN: APPLE_ASAN_INIT_FOR_DLOPEN=0 %env_asan_opts=abort_on_error=0 not \
// RUN:   %run %t %shared_libasan 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-DL-OPEN-FAIL %s
// RUN: env -u APPLE_ASAN_INIT_FOR_DLOPEN %env_asan_opts=abort_on_error=0 not \
// RUN:   %run %t %shared_libasan 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-DL-OPEN-FAIL %s

// Check that we can successfully dlopen the ASan dylib when we set the right
// environment variable.
// RUN: env APPLE_ASAN_INIT_FOR_DLOPEN=1 %run %t %shared_libasan 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-DL-OPEN-SUCCESS %s

#include <dlfcn.h>
#include <stdio.h>

// CHECK-DL-OPEN-FAIL: ERROR: Interceptors are not working

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <dylib_path>\n", argv[0]);
    return 1;
  }
  const char *dylib_path = argv[1];
  void *handle = dlopen(dylib_path, RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "Failed to dlopen: %s\n", dlerror());
    return 1;
  }
  // Make sure we can find a function we expect to be in the dylib.
  void *fn = dlsym(handle, "__sanitizer_mz_size");
  if (!fn) {
    fprintf(stderr, "Failed to get symbol: %s\n", dlerror());
    return 1;
  }
  // TODO(dliew): Actually call a function from the dylib that is safe to call.
  // CHECK-DL-OPEN-SUCCESS: DONE
  printf("DONE\n");
  return 0;
}
