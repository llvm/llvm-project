// RUN: %clangxx -O1 %s -o %t
// RUN: rm -rf %t.tmp
// RUN: %run %t 1
// RUN: %run %t 2

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

void test(const char *path, int flags) {
  int fd = open(path, flags, 0600);
  if (fd == -1) {
    perror(path);
    if (errno == EOPNOTSUPP || errno == EINVAL)
      return;
  }
  assert(fd != -1);
  struct stat info;
  int result = fstat(fd, &info);
  assert((info.st_mode & ~S_IFMT) == 0600);
  assert(result == 0);
  close(fd);
}

int main(int argc, char *argv[]) {
  assert(argc == 2);
  char buff[10000];
  sprintf(buff, "%s.tmp", argv[0]);
  if (atoi(argv[1]) == 1) {
    unlink(buff);
    test(buff, O_RDWR | O_CREAT);
  }

#ifdef O_TMPFILE
  if (atoi(argv[1]) == 2) {
    char *last = strrchr(buff, '/');
    assert(last);
    *last = 0;
    test(buff, O_RDWR | O_TMPFILE);
  }
#endif
}
