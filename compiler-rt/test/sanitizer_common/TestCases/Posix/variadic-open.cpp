// RUN: rm -rf %t.tmp
// RUN: mkdir -p %t.tmp
// RUN: %clangxx -O1 %s -o %t && %run %t %t.tmp/1 %t.tmp

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

void test(const char *path, int flags) {
  assert(path);
  int fd = open(path, flags, 0600);
  if (fd == -1)
    perror ("open");
  assert(fd != -1);
  struct stat info;
  int result = fstat(fd, &info);
  assert((info.st_mode & ~S_IFMT) == 0600);
  assert(result == 0);
  close(fd);
}

int main(int argc, char *argv[]) {
  assert(argc == 3);
  test(argv[1], O_RDWR | O_CREAT);

#ifdef O_TMPFILE
  test(argv[2], O_RDWR | O_TMPFILE);
#endif
}
