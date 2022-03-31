// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

void *Thread(void *arg) {
  char buf;
  read((long)arg, &buf, 1);
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  int fd = open("/dev/random", O_RDONLY);
  pthread_t t;
  pthread_create(&t, NULL, Thread, (void *)(long)fd);
  barrier_wait(&barrier);
  close(fd);
  pthread_join(t, NULL);
  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: DONE
