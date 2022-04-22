// RUN: %clangxx %s -o %t && %run %t %p

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <sys/prctl.h>

#ifndef PR_SCHED_CORE
#  define PR_SCHED_CORE 62
#endif

#ifndef PR_SCHED_CORE_CREATE
#  define PR_SCHED_CORE_CREATE 1
#endif

#ifndef PR_SCHED_CORE_GET
#  define PR_SCHED_CORE_GET 0
#endif

int main() {

  int res;
  res = prctl(PR_SCHED_CORE, PR_SCHED_CORE_CREATE, 0, 0, 0);
  if (res < 0) {
    assert(errno == EINVAL || errno == ENODEV);
    return 0;
  }

  uint64_t cookie = 0;
  res = prctl(PR_SCHED_CORE, PR_SCHED_CORE_GET, 0, 0, &cookie);
  if (res < 0) {
    assert(errno == EINVAL);
  } else {
    assert(cookie != 0);
  }

  return 0;
}
