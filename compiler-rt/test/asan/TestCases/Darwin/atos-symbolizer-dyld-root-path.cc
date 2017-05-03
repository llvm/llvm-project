// Check that when having a DYLD_ROOT_PATH set, the symbolizer still works.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=verbosity=2 DYLD_ROOT_PATH="/" ASAN_SYMBOLIZER_PATH=$(which atos) \
// RUN:   not %run %t 2>&1 | FileCheck %s
//
// Due to a bug in atos, this only works on x86_64.
// REQUIRES: asan-64-bits

// Path returned by `which atos` is invalid on iOS.
// UNSUPPORTED: ios

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(10 * sizeof(char));
  memset(x, 0, 10);
  int res = x[argc];
  free(x);
  free(x + argc - 1);  // BOOM
  // CHECK: Using atos at user-specified path:
  // CHECK: AddressSanitizer: attempting double-free{{.*}}in thread T0
  // CHECK: #0 0x{{.*}} in {{.*}}free
  // CHECK: #1 0x{{.*}} in main {{.*}}atos-symbolizer-dyld-root-path.cc:[[@LINE-4]]
  // CHECK: freed by thread T0 here:
  // CHECK: #0 0x{{.*}} in {{.*}}free
  // CHECK: #1 0x{{.*}} in main {{.*}}atos-symbolizer-dyld-root-path.cc:[[@LINE-8]]
  // CHECK: allocated by thread T0 here:
  // CHECK: atos-symbolizer-dyld-root-path.cc:[[@LINE-13]]
  return res;
}
