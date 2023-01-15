// RUN: rm -rf %t-dir
// RUN: mkdir %t-dir

// RUN: %clangxx_tsan -O1 %s -DLIB -fPIC -shared -o %t-dir/libignore_lib4.so
// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t-dir/executable
// RUN: echo "called_from_lib:libignore_lib4.so" > %t-dir/executable.supp
// RUN: %env_tsan_opts=suppressions='%t-dir/executable.supp' %run %t-dir/executable 2>&1 | FileCheck %s

// powerpc64 big endian bots failed with "FileCheck error: '-' is empty" due
// to a segmentation fault.
// UNSUPPORTED: target=powerpc64-unknown-linux-gnu{{.*}}
// aarch64 bots failed with "called_from_lib suppression 'libignore_lib4.so'
//                           is matched against 2 libraries".
// UNSUPPORTED: target=aarch64{{.*}}

// Test longjmp in ignored lib.
// It used to crash since we jumped out of ScopedInterceptor scope.

#include "test.h"
#include <setjmp.h>
#include <string.h>
#include <errno.h>
#include <libgen.h>
#include <string>

#ifdef LIB

extern "C" void myfunc() {
  for (int i = 0; i < (1 << 20); i++) {
    jmp_buf env;
    if (!setjmp(env))
      longjmp(env, 1);
  }
}

#else

int main(int argc, char **argv) {
  std::string lib = std::string(dirname(argv[0])) + "/libignore_lib4.so";
  void *h = dlopen(lib.c_str(), RTLD_GLOBAL | RTLD_NOW);
  void (*func)() = (void(*)())dlsym(h, "myfunc");
  func();
  fprintf(stderr, "DONE\n");
  return 0;
}

#endif

// CHECK: DONE
