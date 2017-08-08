// RUN: %clangxx -O0 %s -o %t && %run %t
// UNSUPPORTED: lsan

#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>

int main() {
  struct rlimit lim_core;
  getrlimit(RLIMIT_CORE, &lim_core);
  void *p;
  if (sizeof(p) == 8) {
    assert(0 == lim_core.rlim_max);
  }
  return 0;
}
