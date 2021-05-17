// RUN: %check_clang_tidy %s bugprone-signal-handler %t -- -- -isystem %S/Inputs/Headers

#include "signal.h"
#include "stdlib.h"
#include "stdio.h"
#include "system-other.h"

// The function should be classified as system call even if there is
// declaration the in source file.
// FIXME: The detection works only if the first declaration is in system
// header.
int printf(const char *, ...);
typedef void (*sighandler_t)(int);
sighandler_t signal(int signum, sighandler_t handler);

void handler_abort(int) {
  abort();
}

void handler_other(int) {
  printf("1234");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
}

void handler_signal(int) {
  // FIXME: It is only OK to call signal with the current signal number.
  signal(0, SIG_DFL);
}

void f_ok() {
  abort();
}

void f_bad() {
  printf("1234");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
}

void f_extern();

void handler_ok(int) {
  f_ok();
}

void handler_bad(int) {
  f_bad();
}

void handler_extern(int) {
  f_extern();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'f_extern' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
}

void test() {
  signal(SIGINT, handler_abort);
  signal(SIGINT, handler_signal);
  signal(SIGINT, handler_other);

  signal(SIGINT, handler_ok);
  signal(SIGINT, handler_bad);
  signal(SIGINT, handler_extern);

  signal(SIGINT, _Exit);
  signal(SIGINT, other_call);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'other_call' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]

  signal(SIGINT, SIG_IGN);
  signal(SIGINT, SIG_DFL);
}
