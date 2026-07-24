// RUN: %clang -O1 %s -o %t && %run %t
// RUN: %clang -O1 -D_FORTIFY_SOURCE=3 %s -o %t.fortify && %run %t.fortify
// REQUIRES: glibc

#define _LARGEFILE64_SOURCE

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void write_all(int fd, const char *s) {
  size_t n = strlen(s);
  assert(write(fd, s, n) == (ssize_t)n);
}

int main(void) {
  char path[] = "/tmp/sanitizer_common_io_XXXXXX";
  int fd = mkstemp(path);
  assert(fd >= 0);
  unlink(path);

  write_all(fd, "abcdef");
  assert(lseek(fd, 0, SEEK_SET) == 0);

  char buf[8];
  memset(buf, 0, sizeof(buf));
  assert(read(fd, buf, 1) == 1);
  assert(buf[0] == 'a');

  memset(buf, 0, sizeof(buf));
  assert(pread(fd, buf, 1, 2) == 1);
  assert(buf[0] == 'c');

  memset(buf, 0, sizeof(buf));
  assert(pread64(fd, buf, 1, 4) == 1);
  assert(buf[0] == 'e');

  assert(close(fd) == 0);
  return 0;
}
