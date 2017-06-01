// RUN: %clangxx_tsan -O1 %s -DLIB -fPIC -fno-sanitize=thread -shared -o %T/libignore_lib3.so
// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=suppressions='%s.supp' %deflake %run %t | FileCheck %s

// Tests that unloading of a library matched against called_from_lib suppression
// causes program crash (this is not supported).

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
  void *h = dlopen(lib.c_str(), RTLD_GLOBAL | RTLD_NOW);
  dlclose(h);
  fprintf(stderr, "OK\n");
}

#else  // #ifdef LIB

extern "C" void libfunc() {
}

#endif  // #ifdef LIB

// CHECK: ThreadSanitizer: library {{.*}} that was matched against called_from_lib suppression 'ignore_lib3.so' is unloaded
// CHECK-NOT: OK

