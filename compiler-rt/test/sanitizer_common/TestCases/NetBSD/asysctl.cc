// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/param.h>
#include <sys/types.h>

#include <sys/sysctl.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void test_asysctl() {
  int mib[] = {CTL_KERN, KERN_OSTYPE};
  size_t len;
  char *buf = (char *)asysctl(mib, __arraycount(mib), &len);
  assert(buf);

  printf("asysctl: '%s' size: '%zu'\n", buf, len);

  free(buf);
}

void test_asysctlbyname() {
  size_t len;
  char *buf = (char *)asysctlbyname("kern.ostype", &len);
  assert(buf);

  printf("asysctlbyname: '%s' size: '%zu'\n", buf, len);

  free(buf);
}

int main(void) {
  printf("asysctl\n");

  test_asysctl();
  test_asysctlbyname();

  return 0;

  // CHECK: asysctl
  // CHECK: asysctl: '{{.*}}' size: '{{.*}}'
  // CHECK: asysctlbyname: '{{.*}}' size: '{{.*}}'
}
