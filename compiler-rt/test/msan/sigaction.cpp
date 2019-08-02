// RUN: %clangxx_msan -std=c++11 -O0 -g %s -o %t
// RUN: %run %t __
// RUN: not %run %t A_ 2>&1 | FileCheck %s
// RUN: not %run %t AH 2>&1 | FileCheck %s
// RUN: not %run %t B_ 2>&1 | FileCheck %s
// RUN: not %run %t BH 2>&1 | FileCheck %s
// RUN: not %run %t C_ 2>&1 | FileCheck %s
// RUN: not %run %t CH 2>&1 | FileCheck %s

#include <assert.h>
#include <signal.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>

void handler(int) {}
void action(int, siginfo_t *, void *) {}

int main(int argc, char **argv) {
  char T = argv[1][0];
  char H = argv[1][1];
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  if (H == 'H') {
    sa.sa_handler = handler;
  } else {
    sa.sa_sigaction = action;
    sa.sa_flags = SA_SIGINFO;
  }

  if (T == 'A') {
    if (H == 'H')
      __msan_poison(&sa.sa_handler, sizeof(sa.sa_handler));
    else
      __msan_poison(&sa.sa_sigaction, sizeof(sa.sa_sigaction));
  }
  if (T == 'B')
    __msan_poison(&sa.sa_flags, sizeof(sa.sa_flags));
  if (T == 'C')
    __msan_poison(&sa.sa_mask, sizeof(sa.sa_mask));
  // CHECK: use-of-uninitialized-value
  int res = sigaction(SIGUSR1, &sa, nullptr);
  assert(res == 0);
  return 0;
}
