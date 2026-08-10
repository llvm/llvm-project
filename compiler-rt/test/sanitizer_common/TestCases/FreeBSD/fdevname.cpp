// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>

void test_fdevname() {
  int fd = open("/dev/null", O_RDONLY);
  char *name;
  
  printf("test_fdevname\n");
  assert(fd != -1);
  assert((name = fdevname(fd)));
  close(fd);

  printf("%s\n", name);
}

void test_fdevname_r() {
  int fd = open("/dev/null", O_RDONLY);
  char *name;
  char buf[5];

  printf("test_fdevname_r\n");
  assert(fd != -1);
  assert((name = fdevname_r(fd, buf, sizeof(buf))));
  close(fd);

  printf("%s\n", name);
}

int main(void) {
  test_fdevname();
  test_fdevname_r();
  // CHECK: test_fdevname
  // CHECK: null
  // CHECK: test_fdevname_r
  // CHECK: null

  return 0;
}
