// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

int main(void) {
  struct stat st;
  char modep[15];

  if (stat("/etc/hosts", &st))
    exit(1);

  strmode(st.st_mode, modep);

  printf("%s\n", modep);

  return 0;
}
