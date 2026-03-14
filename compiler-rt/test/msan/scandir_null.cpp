// RUN: %clangxx_msan -O0 %s -o %t && %run %t %p
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %p
// RUN: %clangxx_msan -O3 %s -o %t && %run %t %p

#include <assert.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>


int main(int argc, char *argv[]) {
  assert(argc == 2);
  char buf[1024];
  snprintf(buf, sizeof(buf), "%s/%s", argv[1], "scandir_test_root/");
  
  struct dirent **d;
  int res = scandir(buf, &d, NULL, NULL);
  assert(res >= 3);
  assert(__msan_test_shadow(&d, sizeof(*d)) == (size_t)-1);
  for (int i = 0; i < res; ++i) {
    assert(__msan_test_shadow(&d[i], sizeof(d[i])) == (size_t)-1);
    assert(__msan_test_shadow(d[i], d[i]->d_reclen) == (size_t)-1);
  }
  return 0;
}
