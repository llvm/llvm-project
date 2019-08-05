// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/types.h>

#include <sys/statvfs.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  printf("getvfsstat\n");

  int rv = getvfsstat(NULL, 0, ST_WAIT);
  assert(rv != -1);

  size_t sz = rv * sizeof(struct statvfs);
  struct statvfs *buf = (struct statvfs *)malloc(sz);
  assert(buf);

  rv = getvfsstat(buf, sz, ST_WAIT);
  assert(rv != -1);

  for (int i = 0; i < rv; i++) {
    printf("Filesystem %d\n", i);
    printf("\tfstypename=%s\n", buf[i].f_fstypename);
    printf("\tmntonname=%s\n", buf[i].f_mntonname);
    printf("\tmntfromname=%s\n", buf[i].f_mntfromname);
  }

  free(buf);

  // CHECK: getvfsstat

  return 0;
}
