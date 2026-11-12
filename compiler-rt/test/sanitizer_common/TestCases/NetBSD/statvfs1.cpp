// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/param.h>
#include <sys/types.h>

#include <sys/statvfs.h>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void test_statvfs1() {
  printf("statvfs1\n");

  struct statvfs buf;
  int rv = statvfs1("/etc/fstab", &buf, ST_WAIT);
  assert(rv != -1);

  printf("fstypename='%s'\n", buf.f_fstypename);
  printf("mntonname='%s'\n", buf.f_mntonname);
  printf("mntfromname='%s'\n", buf.f_mntfromname);
}

void test_fstatvfs1() {
  printf("fstatvfs1\n");

  int fd = open("/etc/fstab", O_RDONLY);
  assert(fd > 0);

  struct statvfs buf;
  int rv = fstatvfs1(fd, &buf, ST_WAIT);
  assert(rv != -1);

  printf("fstypename='%s'\n", buf.f_fstypename);
  printf("mntonname='%s'\n", buf.f_mntonname);
  printf("mntfromname='%s'\n", buf.f_mntfromname);

  rv = close(fd);
  assert(rv != -1);
}

int main(void) {
  test_statvfs1();
  test_fstatvfs1();

  // CHECK: statvfs1
  // CHECK: fstypename='{{.*}}'
  // CHECK: mntonname='{{.*}}'
  // CHECK: mntfromname='{{.*}}'
  // CHECK: fstatvfs1
  // CHECK: fstypename='{{.*}}'
  // CHECK: mntonname='{{.*}}'
  // CHECK: mntfromname='{{.*}}'

  return 0;
}
