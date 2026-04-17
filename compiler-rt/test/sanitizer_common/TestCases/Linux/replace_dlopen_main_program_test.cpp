// Test 'test_only_replace_dlopen_main_program' flag

// RUN: %clangxx %s -pie -fPIE -o %t
// RUN: env %tool_options='test_only_replace_dlopen_main_program=true' %run %t
// RUN: env %tool_options='test_only_replace_dlopen_main_program=false' not %run %t

// dladdr is 'nonstandard GNU extensions that are also present on Solaris'
// REQUIRES: glibc

// Does not intercept dlopen
// UNSUPPORTED: hwasan, lsan, ubsan

// Flag has no effect with dynamic runtime
// UNSUPPORTED: asan-dynamic-runtime

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// We can't use the address of 'main' (error: ISO C++ does not allow 'main' to be used by a program [-Werror,-Wmain]')
// so we add this function.
__attribute__((noinline, no_sanitize("address"))) void foo() {
  printf("Hello World!\n");
}

int main(int argc, char *argv[]) {
  foo();

  // "If filename is NULL, then the returned handle is for the main program."
  void *correct_handle = dlopen(NULL, RTLD_LAZY);
  printf("dlopen(NULL,...): %p\n", correct_handle);

  Dl_info info;
  if (dladdr((void *)&foo, &info) == 0) {
    printf("dladdr failed\n");
    return 1;
  }
  printf("dladdr(&foo): %s\n", info.dli_fname);
  void *test_handle = dlopen(info.dli_fname, RTLD_LAZY);
  printf("dlopen(%s,...): %p\n", info.dli_fname, test_handle);

  if (test_handle != correct_handle) {
    printf("Error: handles do not match\n");
    return 1;
  }

  return 0;
}
