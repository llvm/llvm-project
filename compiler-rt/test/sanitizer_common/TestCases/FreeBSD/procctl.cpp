// RUN: %clangxx %s -o %t && %run %t %p

#include <assert.h>
#include <errno.h>
#include <unistd.h>
#include <sys/procctl.h>

int main() {
  struct procctl_reaper_status status = {0};
  int res, aslr;
  res = procctl(P_PID, getpid(), PROC_REAP_STATUS, &status);
  if (res < 0) {
    assert(errno == EPERM);
    return 0;
  }

  assert(status.rs_flags >= REAPER_STATUS_OWNED);

  res = procctl(P_PID, getpid(), PROC_ASLR_STATUS, &aslr);
  if (res < 0) {
    assert(errno == EPERM);
    return 0;
  }

  assert(aslr >= PROC_ASLR_FORCE_ENABLE);

  return 0;
}
