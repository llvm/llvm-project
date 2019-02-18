// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, solaris

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

int main(void) {
  printf("sysctl\n");

  test_sysctl();
  test_sysctlbyname();
  test_sysctlnametomib();

  // CHECK: sysctl
  // CHECK: sysctl: '{{.*}}' size: '{{.*}}'
  // CHECK: sysctlbyname: '{{.*}}' size: '{{.*}}'
  // CHECK: sysctlnametomib: '{{.*}}' size: '{{.*}}'

  return 0;
}
