// RUN: rm -rf %t-dir
// RUN: mkdir %t-dir

// RUN: %clangxx_tsan -O1 %s -DLIB -fPIC -fno-sanitize=thread -shared -o %t-dir/libignore_lib2_0.so
// RUN: %clangxx_tsan -O1 %s -DLIB -fPIC -fno-sanitize=thread -shared -o %t-dir/libignore_lib2_1.so
// RUN: %clangxx_tsan -O1 %s -o %t-dir/executable
// RUN: %env_tsan_opts=suppressions='%s.supp' %deflake %run %t-dir/executable | FileCheck %s

// Tests that called_from_lib suppression matched against 2 libraries
// causes program crash (this is not supported).

#ifndef LIB

#include <dlfcn.h>
#include <stdio.h>
#include <libgen.h>
#include <string>

int main(int argc, char **argv) {
  std::string lib0 = std::string(dirname(argv[0])) + "/libignore_lib2_0.so";
  std::string lib1 = std::string(dirname(argv[0])) + "/libignore_lib2_1.so";
  dlopen(lib0.c_str(), RTLD_GLOBAL | RTLD_NOW);
  dlopen(lib1.c_str(), RTLD_GLOBAL | RTLD_NOW);
  fprintf(stderr, "OK\n");
}

#else  // #ifdef LIB

extern "C" void libfunc() {
}

#endif  // #ifdef LIB

// CHECK: ThreadSanitizer: called_from_lib suppression 'ignore_lib2' is matched against 2 libraries
// CHECK-NOT: OK

