// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <stdlib.h>
#include <unistd.h>
#include <grp.h>

int main(void) {
  gid_t *groups;
  gid_t nobody;
  int ngroups;
  int maxgrp;

  maxgrp = sysconf(_SC_NGROUPS_MAX);
  groups = (gid_t *)malloc(maxgrp * sizeof(gid_t));
  if (!groups)
    exit(1);

  if (gid_from_group("nobody", &nobody) == -1)
    exit(1);

  if (getgroupmembership("nobody", nobody, groups, maxgrp, &ngroups))
    exit(1);

  if (groups && ngroups) {
    free(groups);
    exit(0);
  }

  return -1;
}
