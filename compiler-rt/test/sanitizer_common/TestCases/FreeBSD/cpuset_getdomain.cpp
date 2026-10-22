// RUN: %clangxx -O0 %s -o %t && %run %t

#include <assert.h>
#include <sys/cpuset.h>
#include <sys/domainset.h>
#include <sys/types.h>

int main() {
  domainset_t ds;
  int pc;

  int res =
      cpuset_getdomain(CPU_LEVEL_ROOT, CPU_WHICH_PID, -1, sizeof(ds), &ds, &pc);
  assert(res == 0);
  assert(pc != DOMAINSET_POLICY_INVALID);
  return 0;
}
