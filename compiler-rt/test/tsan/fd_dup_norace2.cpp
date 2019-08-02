// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

// dup2(oldfd, newfd) races with read(newfd).
// This is not reported as race because:
// 1. Some software dups a closed pipe in place of a socket before closing
//    the socket (to prevent races actually).
// 2. Some daemons dup /dev/null in place of stdin/stdout.

int fd;

void *Thread(void *x) {
  char buf;
  int n = read(fd, &buf, 1);
  if (n != 1) {
    // This read can "legitimately" fail regadless of the fact that glibc claims
    // that "there is no instant in the middle of calling dup2 at which new is
    // closed and not yet a duplicate of old". Strace of the failing runs
    // looks as follows:
    //
    //    [pid 122196] open("/dev/urandom", O_RDONLY) = 3
    //    [pid 122196] open("/dev/urandom", O_RDONLY) = 4
    //    Process 122382 attached
    //    [pid 122382] read(3,  <unfinished ...>
    //    [pid 122196] dup2(4, 3 <unfinished ...>
    //    [pid 122382] <... read resumed> 0x7fcd139960b7, 1) = -1 EBADF (Bad file descriptor)
    //    [pid 122196] <... dup2 resumed> )       = 3
    //    read failed: n=-1 errno=9
    //
    // The failing read does not interfere with what this test tests,
    // so we just ignore the failure.
    //
    // exit(printf("read failed: n=%d errno=%d\n", n, errno));
  }
  return 0;
}

int main() {
  fd = open("/dev/urandom", O_RDONLY);
  int fd2 = open("/dev/urandom", O_RDONLY);
  if (fd == -1 || fd2 == -1)
    exit(printf("open failed\n"));
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  if (dup2(fd2, fd) == -1)
    exit(printf("dup2 failed\n"));
  pthread_join(th, 0);
  if (close(fd) == -1)
    exit(printf("close failed\n"));
  if (close(fd2) == -1)
    exit(printf("close failed\n"));
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
