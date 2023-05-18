// RUN: %clangxx_msan -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -O1 %s -o %t && %run %t
// RUN: %clangxx_msan -O2 %s -o %t && %run %t
// RUN: %clangxx_msan -O3 %s -o %t && %run %t

// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t
// RUN: %clangxx_msan -O1 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t
// RUN: %clangxx_msan -O2 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t
// RUN: %clangxx_msan -O3 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t

// Test that readdir64 is intercepted as well as readdir.

#include <sys/types.h>
#include <dirent.h>
#include <stdlib.h>


int main(void) {
  DIR *dir = opendir(".");
  struct dirent *d = readdir(dir);
  if (d->d_name[0]) {
    closedir(dir);
    exit(0);
  }
  closedir(dir);
  return 0;
}
