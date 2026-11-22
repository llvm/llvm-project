// Test that calling sigaction() before UBSan initializes still installs
// deadly signal handlers requested via UBSAN_OPTIONS.
//
// RUN: %clangxx -fsanitize=undefined -shared-libsan %s -o %t
// RUN: %env_ubsan_opts=handle_sigill=1 not %run %t 2>&1 | FileCheck %s
//
// REQUIRES: target={{.*linux.*}}, ubsan-standalone
// UNSUPPORTED: android

#include <signal.h>

static void preinit() {
  struct sigaction old;
  sigaction(SIGILL, nullptr, &old);
}

__attribute__((section(".preinit_array"), used)) static void (*p)() = preinit;

int main() {
  // CHECK: ERROR: UndefinedBehaviorSanitizer: ILL
  raise(SIGILL);
}
