// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
// UNSUPPORTED: ios
//
// RUN: %clangxx_asan -O0 %s -o %t && %run %t %p 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && %run %t %p 2>&1 | FileCheck %s

#include <assert.h>
#include <glob.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <string>


int main(int argc, char *argv[]) {
  std::string path = argv[1];
  std::string pattern = path + "/glob_test_root/*a";
  printf("pattern: %s\n", pattern.c_str());

  glob_t globbuf;
  int res = glob(pattern.c_str(), 0, 0, &globbuf);

  printf("%d %s\n", errno, strerror(errno));
  assert(res == 0);
  assert(globbuf.gl_pathc == 2);
  printf("%zu\n", strlen(globbuf.gl_pathv[0]));
  printf("%zu\n", strlen(globbuf.gl_pathv[1]));
  globfree(&globbuf);
  printf("PASS\n");
  // CHECK: PASS
  return 0;
}
