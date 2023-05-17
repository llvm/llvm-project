// RUN: %clangxx -O0 -g %s -o %t && %run %t
//
// REQUIRES: linux || freebsd || netbsd

#include <stdlib.h>
#include <unistd.h>
#include <grp.h>

int main(void) {
  gid_t *groups;
  group *root;
  int ngroups;

  ngroups = sysconf(_SC_NGROUPS_MAX);
  groups = (gid_t *)malloc(ngroups * sizeof(gid_t));
  if (!groups)
    exit(1);

  if (!(root = getgrnam("root")))
    exit(2);

  if (getgrouplist("root", root->gr_gid, groups, &ngroups) == -1)
    exit(3);

  if (groups && ngroups) {
    free(groups);
    exit(0);
  }

  return -1;
}
