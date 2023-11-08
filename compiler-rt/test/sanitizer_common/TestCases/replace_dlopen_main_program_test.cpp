// Test 'test_only_replace_dlopen_main_program' flag

// RUN: %clangxx %s -o %t
// RUN: env %tool_options='test_only_replace_dlopen_main_program=true' %run %t
// RUN: env %tool_options='test_only_replace_dlopen_main_program=false' not %run %t
// REQUIRES: glibc

// Does not intercept dlopen
// UNSUPPORTED: hwasan, lsan, ubsan

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // "If filename is NULL, then the returned handle is for the main program."
  void *correct_handle = dlopen(NULL, RTLD_LAZY);

  // Check that this is equivalent to dlopen(NULL, ...)
  void *handle = dlopen(argv[0], RTLD_LAZY);
  printf("dlopen(NULL,...): %p\n", correct_handle);
  printf("dlopen(<main program>,...): %p\n", handle);
  fflush(stdout);

  if (handle != correct_handle)
    return 1;

  return 0;
}
