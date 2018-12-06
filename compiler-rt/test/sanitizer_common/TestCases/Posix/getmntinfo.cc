// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, darwin, solaris

#include <sys/types.h>

#if defined(__NetBSD__)
#include <sys/statvfs.h>
#else
#include <sys/mount.h>
#endif

#include <err.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  printf("getmntinfo\n");

#if defined(__NetBSD__)
  struct statvfs *fss;
#else
  struct statfs *fss;
#endif
  int nfss = getmntinfo(&fss, MNT_NOWAIT);
  if (nfss <= 0)
    errx(1, "getmntinfo");

  for (int i = 0; i < nfss; i++)
    printf("%d: %s\n", i, fss[i].f_fstypename);

  // CHECK: getmntinfo

  return 0;
}
