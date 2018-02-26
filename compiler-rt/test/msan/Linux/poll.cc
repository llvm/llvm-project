// RUN: %clangxx_msan -O0 -std=c++11 -g %s -o %t
// RUN: %run %t _ 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: not %run %t A 2>&1 | FileCheck %s --check-prefix=A
// RUN: not %run %t B 2>&1 | FileCheck %s --check-prefix=B

#include <assert.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char **argv) {
  char T = argv[1][0];

  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 1000;
  int res = ppoll(nullptr, 0, &ts, nullptr);
  assert(res == 0);

  if (T == 'A') {
    __msan_poison(&ts.tv_sec, sizeof(ts.tv_sec));
    ppoll(nullptr, 0, &ts, nullptr);
    // A: use-of-uninitialized-value
  }

  // A-NOT: ==1
  // B: ==1
  fprintf(stderr, "==1\n");

  sigset_t sig;
  if (T != 'B')
    sigemptyset(&sig);
  ppoll(nullptr, 0, &ts, &sig);
  // B: use-of-uninitialized-value

  // B-NOT: ==2
  // CLEAN: ==2
  fprintf(stderr, "==2\n");
  return 0;
}
