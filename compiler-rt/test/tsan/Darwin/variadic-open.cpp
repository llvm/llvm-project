// RUN: %clangxx_tsan -O1 %s -o %t && %run %t %t.tmp 2>&1 | FileCheck %s
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  assert(argv[1]);
  unlink(argv[1]);
  int fd = open(argv[1], O_RDWR | O_CREAT, 0600);
  assert(fd != -1);
  struct stat info;
  int result = fstat(fd, &info);
  fprintf(stderr, "permissions = 0%o\n", info.st_mode & ~S_IFMT);
  assert(result == 0);
  close(fd);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: permissions = 0600
// CHECK: Done.
