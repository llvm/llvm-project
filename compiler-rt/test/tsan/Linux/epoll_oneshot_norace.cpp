// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// EPOLL_CTL_MOD rearms an EPOLLONESHOT fd.  The epoll_wait that observes the
// rearmed, already-ready fd synchronizes with the thread that rearmed it.

#include "../test.h"
#include <atomic>
#include <errno.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>

int efd;
int fd;
int data;
std::atomic<bool> done;

int main() {
  efd = epoll_create(1);
  if (efd == -1)
    exit(printf("epoll_create failed: %d\n", errno));
  fd = eventfd(1, 0);
  if (fd == -1)
    exit(printf("eventfd failed: %d\n", errno));

  epoll_event event = {.events = EPOLLIN | EPOLLONESHOT};
  if (epoll_ctl(efd, EPOLL_CTL_ADD, fd, &event))
    exit(printf("epoll_ctl add failed: %d\n", errno));

  epoll_event events[1] = {};
  if (epoll_wait(efd, events, 1, -1) != 1)
    exit(printf("first epoll_wait failed: %d\n", errno));

  pthread_t th;
  pthread_create(
      &th, nullptr,
      +[](void *) -> void * {
        data = 42;
        epoll_event event = {.events = EPOLLIN | EPOLLONESHOT};
        if (epoll_ctl(efd, EPOLL_CTL_MOD, fd, &event))
          exit(printf("epoll_ctl mod failed: %d\n", errno));
        done.store(true, std::memory_order_relaxed);
        return nullptr;
      },
      nullptr);

  if (epoll_wait(efd, events, 1, -1) != 1)
    exit(printf("second epoll_wait failed: %d\n", errno));
  while (!done.load(std::memory_order_relaxed))
    sched_yield();
  fprintf(stderr, "data = %d\n", data);

  pthread_join(th, nullptr);
  close(fd);
  close(efd);
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: data = 42
// CHECK-NOT: WARNING: ThreadSanitizer: data race
