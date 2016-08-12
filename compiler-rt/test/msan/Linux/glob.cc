// RUN: %clangxx_msan -O0 %s -o %t && %run %t %p 2>&1 | FileCheck %s
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %p 2>&1 | FileCheck %s
// RUN: %clangxx_msan -O3 %s -o %t && %run %t %p 2>&1 | FileCheck %s

// XFAIL: target-is-mips64el

#include <assert.h>
#include <glob.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

int main(int argc, char *argv[]) {
  assert(argc == 2);
  char buf[1024];
  snprintf(buf, sizeof(buf), "%s/%s", argv[1], "glob_test_root/*a");

  glob_t globbuf;
  int res = glob(buf, 0, 0, &globbuf);

  printf("%d %s\n", errno, strerror(errno));
  assert(res == 0);
  assert(globbuf.gl_pathc == 2);
  printf("%zu\n", strlen(globbuf.gl_pathv[0]));
  printf("%zu\n", strlen(globbuf.gl_pathv[1]));
  printf("PASS\n");
  // CHECK: PASS
  return 0;
}
