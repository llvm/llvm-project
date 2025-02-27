// RUN: %clangxx -O0 %s -o %t

// REQUIRES: glibc

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#if !defined(__GLIBC_PREREQ)
#  define __GLIBC_PREREQ(a, b) 0
#endif

#if !__GLIBC_PREREQ(2, 27)
#  define copy_file_range(a, b, c, d, e)                                       \
    (ssize_t) syscall(__NR_copy_file_range, a, b, c, d, e)
#endif

int main(void) {
  int fdin = open("/proc/self/maps", O_RDONLY);
  assert(fdin > 0);
  char tmp[] = "/tmp/map.XXXXXX";
  int fdout = mkstemp(tmp);
  assert(fdout > 0);
  off_t offin = -1, offout = 0;
  ssize_t cpy = copy_file_range(fdin, &offin, fdout, &offout, 8, 0);
  assert(cpy < 0);
  offin = 0;
  offout = 16;
  cpy = copy_file_range(fdin, &offin, fdout, &offout, 8, 0);
  assert(cpy < 0);
  offout = 0;
  cpy = copy_file_range(fdin, &offin, fdout, &offout, 8, 0);
  assert(cpy == 8);
  close(fdout);
  close(fdin);
  return 0;
}
