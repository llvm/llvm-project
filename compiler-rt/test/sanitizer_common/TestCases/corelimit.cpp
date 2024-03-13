// RUN: %clangxx -O0 %s -o %t && %run %t
// UNSUPPORTED: lsan,ubsan

#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>

int main() {
  struct rlimit lim_core;
  getrlimit(RLIMIT_CORE, &lim_core);
  void *p;
  if (sizeof(p) == 8) {
#ifdef __linux__
    // See comments in DisableCoreDumperIfNecessary().
    assert(lim_core.rlim_cur == 1);
#else
    assert(lim_core.rlim_cur == 0);
#endif
  }
  return 0;
}
