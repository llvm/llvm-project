// RUN: %clangxx_msan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <dlfcn.h>
#include <stdlib.h>

static int my_global;

int main(void) {
  int *uninit = (int*)malloc(sizeof(int));
  my_global = *uninit;
  void *p = dlopen(0, RTLD_NOW);
  assert(p && "failed to get handle to executable");
  return my_global;
  // CHECK: MemorySanitizer: use-of-uninitialized-value
  // CHECK: #0 {{.*}} in main{{.*}}dlopen_executable.cc:[[@LINE-2]]
}
