// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/param.h>
#include <sys/types.h>

#include <sys/sysctl.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void test_sysctlgetmibinfo() {
  int mib[CTL_MAXNAME];
  unsigned int mib_len = __arraycount(mib);
  int rv = sysctlgetmibinfo("kern.ostype", &mib[0], &mib_len, NULL, NULL, NULL,
                       SYSCTL_VERSION);
  assert(!rv);

  char buf[100];
  size_t len = sizeof(buf);
  rv = sysctl(mib, mib_len, buf, &len, NULL, 0);
  assert(!rv);

  printf("sysctlgetmibinfo: '%s' size: '%zu'\n", buf, len);
}

int main(void) {
  printf("sysctlgetmibinfo\n");

  test_sysctlgetmibinfo();

  return 0;

  // CHECK: sysctlgetmibinfo
  // CHECK: sysctlgetmibinfo: '{{.*}}' size: '{{.*}}'
}
