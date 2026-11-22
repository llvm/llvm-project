// Test that calling sigaction() before shared UBSan initializes still respects
// handle_sigill=0, 1, 2 in UBSAN_OPTIONS.
//
// RUN: %clangxx -fsanitize=undefined -shared-libsan %s -o %t-query
// RUN: %env_ubsan_opts=handle_sigill=0 not --crash %run %t-query
// RUN: %env_ubsan_opts=handle_sigill=1 not %run %t-query 2>&1 | FileCheck --check-prefix=UBSAN %s
// RUN: %env_ubsan_opts=handle_sigill=2 not %run %t-query 2>&1 | FileCheck --check-prefix=UBSAN %s
//
// RUN: %clangxx -fsanitize=undefined -shared-libsan -DINSTALL_HANDLER %s -o %t-install
// RUN: %env_ubsan_opts=handle_sigill=0 %run %t-install 2>&1 | FileCheck --check-prefix=USER %s
// RUN: %env_ubsan_opts=handle_sigill=1 %run %t-install 2>&1 | FileCheck --check-prefix=USER %s
// RUN: %env_ubsan_opts=handle_sigill=2 not %run %t-install 2>&1 | FileCheck --check-prefix=UBSAN %s
//
// REQUIRES: target={{.*linux.*}}, ubsan-standalone-shared
// UNSUPPORTED: android

#include <signal.h>
#include <unistd.h>

#ifdef INSTALL_HANDLER
static void user_handler(int) {
  (void)!write(2, "USER_HANDLER\n", 13);
  _exit(0);
}
#endif

static void preinit() {
#ifdef INSTALL_HANDLER
  struct sigaction act = {};
  act.sa_handler = user_handler;
  sigaction(SIGILL, &act, nullptr);
#else
  struct sigaction old;
  sigaction(SIGILL, nullptr, &old);
#endif
}

__attribute__((section(".preinit_array"), used)) static void (*p)() = preinit;

int main() {
  // UBSAN: ERROR: UndefinedBehaviorSanitizer: ILL
  // USER: USER_HANDLER
  raise(SIGILL);
}
