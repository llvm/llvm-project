// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// The test captures what some high-performance networking servers do.
// One thread writes to an fd, and another just receives an epoll
// notification about the write to synchronize with the first thread
// w/o actually reading from the fd.

#include "../test.h"
#include <errno.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>

int main() {
  int efd = epoll_create(1);
  if (efd == -1)
    exit(printf("epoll_create failed: %d\n", errno));
  int fd = eventfd(0, 0);
  if (fd == -1)
    exit(printf("eventfd failed: %d\n", errno));
  epoll_event event = {.events = EPOLLIN | EPOLLET};
  if (epoll_ctl(efd, EPOLL_CTL_ADD, fd, &event))
    exit(printf("epoll_ctl failed: %d\n", errno));
  pthread_t th;
  pthread_create(
      &th, nullptr,
      +[](void *arg) -> void * {
        long long to_add = 1;
        if (write((long)arg, &to_add, sizeof(to_add)) != sizeof(to_add))
          exit(printf("write failed: %d\n", errno));
        return nullptr;
      },
      (void *)(long)fd);
  struct epoll_event events[1] = {};
  if (epoll_wait(efd, events, 1, -1) != 1)
    exit(printf("epoll_wait failed: %d\n", errno));
  close(fd);
  pthread_join(th, nullptr);
  close(efd);
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
