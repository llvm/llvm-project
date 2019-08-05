// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <pwd.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  const char *nobody;

  if (!(nobody = user_from_uid(0, 0)))
    exit(1);

  if (strlen(nobody))
    exit(0);

  return 0;
}
