// RUN: %clangxx %s -o %t && %run %t %p

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <sys/prctl.h>

int main() {

  int res;
  res = prctl(PR_SCHED_CORE, PR_SCHED_CORE_CREATE, 0, 0, 0);
  if (res < 0) {
    assert(errno == EINVAL);
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
