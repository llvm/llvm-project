// RUN: rm -rf %t-dir
// RUN: mkdir %t-dir

// RUN: %clangxx_tsan -O1 -fno-builtin %s -DLIB -fPIC -fno-sanitize=thread -shared -o %t-dir/libignore_lib1.so
// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t-dir/executable
// RUN: echo running w/o suppressions:
// RUN: %deflake %run %t-dir/executable | FileCheck %s --check-prefix=CHECK-NOSUPP
// RUN: echo running with suppressions:
// RUN: %env_tsan_opts=suppressions='%s.supp' %run %t-dir/executable 2>&1 | FileCheck %s --check-prefix=CHECK-WITHSUPP

// Tests that interceptors coming from a dynamically loaded library specified
// in called_from_lib suppression are ignored.

// REQUIRES: stable-runtime
// UNSUPPORTED: powerpc64le
// FIXME: This test regularly fails on powerpc64 LE possibly starting with
// r279664.  Re-enable the test once the problem(s) have been fixed.

#ifndef LIB

#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <libgen.h>
#include <string>

int main(int argc, char **argv) {
  std::string lib = std::string(dirname(argv[0])) + "/libignore_lib1.so";
  void *h = dlopen(lib.c_str(), RTLD_GLOBAL | RTLD_NOW);
  if (h == 0)
    exit(printf("failed to load the library (%d)\n", errno));
  void (*f)() = (void(*)())dlsym(h, "libfunc");
  if (f == 0)
    exit(printf("failed to find the func (%d)\n", errno));
  f();
}

#else  // #ifdef LIB

#include "ignore_lib_lib.h"

#endif  // #ifdef LIB

// CHECK-NOSUPP: WARNING: ThreadSanitizer: data race
// CHECK-NOSUPP: OK

// CHECK-WITHSUPP-NOT: WARNING: ThreadSanitizer: data race
// CHECK-WITHSUPP: OK

