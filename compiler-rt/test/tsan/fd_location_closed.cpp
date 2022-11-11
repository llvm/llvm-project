// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

#include <fcntl.h>

void *Thread(void *x) {
  int fd = (long)x;
  char buf;
  read(fd, &buf, 1);
  barrier_wait(&barrier);
  close(fd);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  int fd = open("/dev/random", O_RDONLY);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread, (void *)(long)fd);
  pthread_create(&t[1], NULL, Thread, (void *)(long)fd);

  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Location is file descriptor {{[0-9]+}} {{(destroyed by thread|created by main)}} 
// CHECK:     #0 {{close|open}}
// CHECK:     #1 {{Thread|main}}
