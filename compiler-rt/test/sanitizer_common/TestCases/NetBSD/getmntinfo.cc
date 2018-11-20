// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/types.h>

#include <sys/statvfs.h>

#include <err.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  printf("getmntinfo\n");

  struct statvfs *fss;
  int nfss = getmntinfo(&fss, MNT_NOWAIT);
  if (nfss <= 0)
    errx(1, "getmntinfo");

  for (int i = 0; i < nfss; i++)
    printf("%d: %s\n", i, fss[i].f_fstypename);

  // CHECK: getmntinfo

  return 0;
}
