// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <grp.h>
#include <stdlib.h>

int main(void) {
  gid_t nobody;

  if (gid_from_group("nobody", &nobody) == -1)
    exit(1);

  if (nobody)
    exit(0);

  return 0;
}
