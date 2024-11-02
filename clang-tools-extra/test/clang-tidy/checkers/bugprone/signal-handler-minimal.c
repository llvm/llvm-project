// RUN: %check_clang_tidy %s bugprone-signal-handler %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: bugprone-signal-handler.AsyncSafeFunctionSet, value: "minimal"}]}' \
// RUN: -- -isystem %clang_tidy_headers

#include "signal.h"
#include "stdlib.h"
#include "string.h"
#include "unistd.h"

void handler_bad1(int) {
  _exit(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: standard function '_exit' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
}

void handler_bad2(int) {
  void *dst;
  const void *src;
  memcpy(dst, src, 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: standard function 'memcpy' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
}

void handler_good(int) {
  abort();
  _Exit(0);
  quick_exit(0);
  signal(0, SIG_DFL);
}

void test(void) {
  signal(SIGINT, handler_bad1);
  signal(SIGINT, handler_bad2);
  signal(SIGINT, handler_good);
}
