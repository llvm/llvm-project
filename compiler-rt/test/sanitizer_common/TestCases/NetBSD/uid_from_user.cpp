// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <pwd.h>
#include <stdlib.h>

int main(void) {
  uid_t nobody;

  if (uid_from_user("nobody", &nobody) == -1)
    exit(1);

  if (nobody)
    exit(0);

  return 0;
}
