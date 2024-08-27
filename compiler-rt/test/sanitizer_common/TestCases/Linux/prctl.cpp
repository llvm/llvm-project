// RUN: %clangxx %s -o %t && %run %t %p

// UNSUPPORTED: android

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
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

#ifndef PR_SET_VMA
#  define PR_SET_VMA 0x53564d41
#  define PR_SET_VMA_ANON_NAME 0
#endif

int main() {
  int res;
  res = prctl(PR_SCHED_CORE, PR_SCHED_CORE_CREATE, 0, 0, 0);
  if (res < 0) {
    assert(errno == EINVAL || errno == ENODEV);
  } else {
    uint64_t cookie = 0;
    res = prctl(PR_SCHED_CORE, PR_SCHED_CORE_GET, 0, 0, &cookie);
    if (res < 0) {
      assert(errno == EINVAL);
    } else {
      assert(cookie != 0);
    }
  }

  int signum;
  res = prctl(PR_GET_PDEATHSIG, reinterpret_cast<unsigned long>(&signum));
  if (res < 0) {
    assert(errno == EINVAL);
  } else {
    assert(signum == 0);
  }

  char invname[81], vlname[] = "prctl";
  for (auto i = 0; i < sizeof(invname); i++) {
    invname[i] = 0x1e;
  }
  invname[80] = 0;
  auto p =
      mmap(nullptr, 128, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  assert(p != MAP_FAILED);
  // regardless of kernel support, the name is invalid
  res = prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, (uintptr_t)p, 128,
              (uintptr_t)invname);
  assert(res == -1);
  assert(errno == EINVAL);
  res = prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, (uintptr_t)p, 128,
              (uintptr_t)vlname);
  if (res < 0) {
    assert(errno == EINVAL);
  }
  munmap(p, 128);

  res = prctl(PR_SET_NAME, "tname");
  if (res == 0) {
    char name[16];
    res = prctl(PR_GET_NAME, name);
    if (res == 0) {
      assert(!strcmp(name, "tname"));
    }
  }

  return 0;
}
