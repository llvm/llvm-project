// RUN: %clangxx_tsan -O1 %s -DBUILD_SO -fPIC -shared -o %t-so.so
// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t && %run %t 2>&1 | FileCheck %s

// Test case for
// https://github.com/google/sanitizers/issues/487

#ifdef BUILD_SO

#include <stdio.h>

extern "C"
void sofunc() {
  fprintf(stderr, "HELLO FROM SO\n");
}

#else  // BUILD_SO

#include <dlfcn.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <string>

void *lib;
void *lib2;

struct Closer {
  ~Closer() {
    dlclose(lib);
    fprintf(stderr, "CLOSED SO\n");
  }
};
static Closer c;

int main(int argc, char *argv[]) {
  lib = dlopen((std::string(argv[0]) + std::string("-so.so")).c_str(),
      RTLD_NOW|RTLD_NODELETE);
  if (lib == 0) {
    printf("error in dlopen: %s\n", dlerror());
    return 1;
  }
  void *f = dlsym(lib, "sofunc");
  if (f == 0) {
    printf("error in dlsym: %s\n", dlerror());
    return 1;
  }
  ((void(*)())f)();
  return 0;
}

#endif  // BUILD_SO

// CHECK: HELLO FROM SO
// CHECK-NOT: Inconsistency detected by ld.so
// CHECK: CLOSED SO

