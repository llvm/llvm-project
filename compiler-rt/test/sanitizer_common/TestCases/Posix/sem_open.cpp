// RUN: %clangxx -O0 %s -o %t && %run %t

// Android does not implement these calls.
// UNSUPPORTED: android

#include <assert.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>

int main() {
  char name[1024];
  sprintf(name, "/sem_open_test_%zu", (size_t)getpid());

  sem_t *s = sem_open(name, O_CREAT, 0644, 123);
  assert(s != SEM_FAILED);
  assert(sem_close(s) == 0);
  assert(sem_unlink(name) == 0);
}
