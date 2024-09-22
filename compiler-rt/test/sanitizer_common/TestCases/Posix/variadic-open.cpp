// RUN: %clangxx -O1 %s -o %t && %run %t %t.tmp

#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
  assert(argv[1]);
  unlink(argv[1]);
  int fd = open(argv[1], O_RDWR | O_CREAT, 0600);
  assert(fd != -1);
  struct stat info;
  int result = fstat(fd, &info);
  assert((info.st_mode & ~S_IFMT) == 0600);
  assert(result == 0);
  close(fd);
}
