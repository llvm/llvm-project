// RUN: rm -rf %t-dir
// RUN: mkdir %t-dir

// RUN: %clangxx_tsan -O1 %s -DLIB -fPIC -fno-sanitize=thread -shared -o %t-dir/libignore_lib3.so
// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t-dir/executable
// RUN: %env_tsan_opts=suppressions='%s.supp':verbosity=1 %run %t-dir/executable 2>&1 | FileCheck %s

// Tests that unloading of a library matched against called_from_lib suppression
// is supported.

// Some aarch64 kernels do not support non executable write pages
// REQUIRES: stable-runtime

#ifndef LIB

#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <libgen.h>
#include <string>

int main(int argc, char **argv) {
  std::string lib = std::string(dirname(argv[0])) + "/libignore_lib3.so";
  void *h;
  void (*f)();
  // Try opening, closing and reopening the ignored lib.
  for (unsigned int k = 0; k < 2; k++) {
    h = dlopen(lib.c_str(), RTLD_GLOBAL | RTLD_NOW);
    if (h == 0)
      exit(printf("failed to load the library (%d)\n", errno));
    f = (void (*)())dlsym(h, "libfunc");
    if (f == 0)
      exit(printf("failed to find the func (%d)\n", errno));
    f();
    dlclose(h);
  }
  fprintf(stderr, "OK\n");
}

#else  // #ifdef LIB

#  include "ignore_lib_lib.h"

#endif  // #ifdef LIB

// CHECK: Matched called_from_lib suppression 'ignore_lib3.so'
// CHECK: library '{{.*}}ignore_lib3.so' that was matched against called_from_lib suppression 'ignore_lib3.so' is unloaded
// CHECK: Matched called_from_lib suppression 'ignore_lib3.so'
// CHECK: library '{{.*}}ignore_lib3.so' that was matched against called_from_lib suppression 'ignore_lib3.so' is unloaded
// CHECK: OK
