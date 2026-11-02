// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int value;
int duplicate_write_fd;

void *writer(void *) {
  value = 1;
  if (write(duplicate_write_fd, ".", 1) != 1)
    abort();
  return nullptr;
}

int main() {
  int pipe_fds[2];
  if (pipe2(pipe_fds, O_CLOEXEC) != 0)
    abort();

  duplicate_write_fd = fcntl(pipe_fds[1], F_DUPFD_CLOEXEC, 0);
  if (duplicate_write_fd < 0)
    abort();

  pthread_t thread;
  pthread_create(&thread, nullptr, writer, nullptr);

  char byte;
  if (read(pipe_fds[0], &byte, 1) != 1)
    abort();
  if (value != 1)
    abort();

  pthread_join(thread, nullptr);
  close(duplicate_write_fd);
  close(pipe_fds[0]);
  close(pipe_fds[1]);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
