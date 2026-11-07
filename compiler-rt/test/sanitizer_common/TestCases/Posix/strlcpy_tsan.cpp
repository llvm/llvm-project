// RUN: %clangxx -O0 -g %s -o %t
// RUN: not %run %t cpy 2>&1 | FileCheck %s --check-prefix=CPY
// RUN: not %run %t cat 2>&1 | FileCheck %s --check-prefix=CAT

// REQUIRES: tsan
// UNSUPPORTED: target={{.*linux.*}} && !glibc-2.38, target={{.*solaris.*}}

#include <pthread.h>
#include <string.h>

#include "tsan/test.h"

static char buf[8] = "ab";
static volatile char sink;

static void *reader_thread(void *arg) {
  sink = buf[3];
  barrier_wait(&barrier);
  return nullptr;
}

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, nullptr, reader_thread, nullptr);
  barrier_wait(&barrier);

  size_t res = 0;
  if (strcmp(argv[1], "cpy") == 0) {
    // CPY: WARNING: ThreadSanitizer: data race
    // CPY: Write of size 4 at {{.*}} by main thread:
    // CPY: #0 {{(wrap_|___interceptor_)?}}strlcpy
    res = strlcpy(buf, "abc", sizeof(buf));
  } else if (strcmp(argv[1], "cat") == 0) {
    // CAT: WARNING: ThreadSanitizer: data race
    // CAT: Write of size 3 at {{.*}} by main thread:
    // CAT: #0 {{(wrap_|___interceptor_)?}}strlcat
    res = strlcat(buf, "cd", sizeof(buf));
  }

  pthread_join(th, nullptr);
  return res == 99;
}
