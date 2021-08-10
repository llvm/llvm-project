// RUN: %clangxx -O0 %s -o %t && %run %t

// Android does not implement this calls.
// UNSUPPORTED: android

#include <assert.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>

int main() {
  char name[1024];
  sprintf(name, "/sem_open_test_%d", getpid());

  sem_t *s1 = sem_open(name, O_CREAT, 0644, 123);
  assert(s1 != SEM_FAILED);

  sem_t *s2 = sem_open(name, O_CREAT, 0644, 123);
  assert(s2 != SEM_FAILED);

  assert(sem_close(s1) == 0);
  assert(sem_close(s2) == 0);

  assert(sem_unlink(name) == 0);
}
