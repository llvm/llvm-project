// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, solaris

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <fstab.h>

int main(void) {
  printf("getfsent\n");

  setfsent();
  struct fstab *fentry = getfsent();

  assert(fentry);

  setfsent();
  struct fstab *pentry = getfsspec(fentry->fs_spec);
  assert(pentry);
  setfsent();
  struct fstab *wentry = getfsfile(fentry->fs_file);
  assert(wentry);
  assert(!memcmp(fentry, wentry, sizeof(*wentry)));
  assert(!memcmp(pentry, wentry, sizeof(*pentry)));

  printf("First entry: device block '%s', mounted with '%s'\n",
    fentry->fs_spec, fentry->fs_mntops);

  endfsent();

  return 0;
  // CHECK: getfsent
  // CHECK: First entry: device block '{{.*}}', mounted with '{{.*}}'
}
