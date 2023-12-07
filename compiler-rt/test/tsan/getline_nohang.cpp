// RUN: %clangxx_tsan -O1 %s -o %t && %run %t

// Data race randomly triggered.
// UNSUPPORTED: target={{.*netbsd.*}}

// Make sure TSan doesn't deadlock on a file stream lock at program shutdown.
// See https://github.com/google/sanitizers/issues/454
#ifdef __FreeBSD__
#define _WITH_GETLINE  // to declare getline()
#endif

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void *thread(void *unused) {
  char *line = NULL;
  size_t size;
  int fd[2];
  pipe(fd);
  // Forge a non-standard stream to make sure it's not closed.
  FILE *stream = fdopen(fd[0], "r");
  while (1) {
    volatile int res = getline(&line, &size, stream);
    (void)res;
  }
  return NULL;
}

int main() {
  pthread_t t;
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
  pthread_create(&t, &a, thread, NULL);
  pthread_attr_destroy(&a);
  fprintf(stderr, "DONE\n");
  return 0;
  // ThreadSanitizer used to hang here because of a deadlock on a file stream.
}

// CHECK: DONE
