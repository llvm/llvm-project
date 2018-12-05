// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, darwin, solaris

#include <sys/param.h>
#include <sys/types.h>

#include <sys/sysctl.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef __arraycount
#define __arraycount(a) (sizeof(a) / sizeof(a[0]))
#endif

void test_sysctl() {
  char buf[100];
  size_t len = sizeof(buf);
  int mib[] = {CTL_KERN, KERN_OSTYPE};
  int rv = sysctl(mib, __arraycount(mib), buf, &len, NULL, 0);
  assert(!rv);

  printf("sysctl: '%s' size: '%zu'\n", buf, len);
}

void test_sysctlbyname() {
  char buf[100];
  size_t len = sizeof(buf);
  int rv = sysctlbyname("kern.ostype", buf, &len, NULL, 0);
  assert(!rv);

  printf("sysctlbyname: '%s' size: '%zu'\n", buf, len);
}

void test_sysctlnametomib() {
  int mib[CTL_MAXNAME];
  size_t mib_len = __arraycount(mib);
  int rv = sysctlnametomib("kern.ostype", &mib[0], &mib_len);
  assert(!rv);

  char buf[100];
  size_t len = sizeof(buf);
  rv = sysctl(mib, mib_len, buf, &len, NULL, 0);
  assert(!rv);

  printf("sysctlnametomib: '%s' size: '%zu'\n", buf, len);
}

#if defined(__NetBSD__)
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
#endif

int main(void) {
  printf("sysctl\n");

  test_sysctl();
  test_sysctlbyname();
  test_sysctlnametomib();
#if defined(__NetBSD__)
  test_asysctl();
  test_asysctlbyname();
  test_sysctlgetmibinfo();
#endif

  // CHECK: sysctl
  // CHECK: sysctl: '{{.*}}' size: '{{.*}}'
  // CHECK: sysctlbyname: '{{.*}}' size: '{{.*}}'
  // CHECK: sysctlnametomib: '{{.*}}' size: '{{.*}}'
#if defined(__NetBSD__)
  // CHECK: asysctl: '{{.*}}' size: '{{.*}}'
  // CHECK: asysctlbyname: '{{.*}}' size: '{{.*}}'
  // CHECK: sysctlgetmibinfo: '{{.*}}' size: '{{.*}}'
#endif

  return 0;
}
