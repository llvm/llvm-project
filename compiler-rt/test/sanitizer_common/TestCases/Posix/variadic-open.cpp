// RUN: %clangxx -O1 %s -o %t && %run %t %t.tmp %T

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

void test(const char *path, int flags) {
  int fd = open(path, flags, 0600);
  if (fd == -1)
    perror(path);
  assert(fd != -1);
  struct stat info;
  int result = fstat(fd, &info);
  assert((info.st_mode & ~S_IFMT) == 0600);
  assert(result == 0);
  close(fd);
}

int main(int argc, char *argv[]) {
  assert(argc == 3);
  assert(argv[1]);
  unlink(argv[1]);
  test(argv[1], O_RDWR | O_CREAT);

#ifdef O_TMPFILE
assert(argv[2]);
  test(argv[2], O_RDWR | O_TMPFILE);
#endif
}
