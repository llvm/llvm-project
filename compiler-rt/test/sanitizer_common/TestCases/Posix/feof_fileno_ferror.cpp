// RUN: %clangxx -g %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
  FILE *fp = fopen(argv[0], "r");
  assert(fp);

  // file should be good upon opening
  assert(!feof(fp) && !ferror(fp));

  // read until EOF
  char buf[BUFSIZ];
  while (fread(buf, 1, sizeof buf, fp) != 0) {}
  assert(feof(fp));

  // clear EOF
  clearerr(fp);
  assert(!feof(fp) && !ferror(fp));

  // get file descriptor
  int fd = fileno(fp);
  assert(fd != -1);

  // break the file by closing underlying descriptor
  assert(close(fd) != -1);

  // verify that an error is signalled
  assert(fread(buf, 1, sizeof buf, fp) == 0);
  assert(ferror(fp));

  // clear error
  clearerr(fp);
  assert(!feof(fp) && !ferror(fp));

  // fclose() will return EBADF because of closed fd
  assert(fclose(fp) == -1);
  return 0;
}
