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

void f_extern(void);

void handler_printf(int) {
  printf("1234");
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'printf' called here from 'handler_printf'
  // CHECK-NOTES: :[[@LINE+4]]:18: note: function 'handler_printf' registered here as signal handler
}

void test_printf(void) {
  signal(SIGINT, handler_printf);
}

void handler_extern(int) {
  f_extern();
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'f_extern' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'f_extern' called here from 'handler_extern'
  // CHECK-NOTES: :[[@LINE+4]]:18: note: function 'handler_extern' registered here as signal handler
}

void test_extern(void) {
  signal(SIGINT, handler_extern);
}

void f_ok(void) {
  abort();
}

void handler_ok(int) {
  f_ok();
}

void test_ok(void) {
  signal(SIGINT, handler_ok);
}

void f_bad(void) {
  printf("1234");
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'printf' called here from 'f_bad'
  // CHECK-NOTES: :[[@LINE+5]]:3: note: function 'f_bad' called here from 'handler_bad'
  // CHECK-NOTES: :[[@LINE+8]]:18: note: function 'handler_bad' registered here as signal handler
}

void handler_bad(int) {
  f_bad();
}

void test_bad(void) {
  signal(SIGINT, handler_bad);
}

void f_bad1(void) {
  printf("1234");
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'printf' called here from 'f_bad1'
  // CHECK-NOTES: :[[@LINE+6]]:3: note: function 'f_bad1' called here from 'f_bad2'
  // CHECK-NOTES: :[[@LINE+9]]:3: note: function 'f_bad2' called here from 'handler_bad1'
  // CHECK-NOTES: :[[@LINE+13]]:18: note: function 'handler_bad1' registered here as signal handler
}

void f_bad2(void) {
  f_bad1();
}

void handler_bad1(int) {
  f_bad2();
  f_bad1();
}

void test_bad1(void) {
  signal(SIGINT, handler_bad1);
}

void handler_abort(int) {
  abort();
}

void handler_signal(int) {
  // FIXME: It is only OK to call signal with the current signal number.
  signal(0, SIG_DFL);
}

void handler_false_condition(int) {
  if (0)
    printf("1234");
  // CHECK-NOTES: :[[@LINE-1]]:5: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:5: note: function 'printf' called here from 'handler_false_condition'
  // CHECK-NOTES: :[[@LINE+4]]:18: note: function 'handler_false_condition' registered here as signal handler
}

void test_false_condition(void) {
  signal(SIGINT, handler_false_condition);
}

void handler_multiple_calls(int) {
  f_extern();
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'f_extern' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'f_extern' called here from 'handler_multiple_calls'
  // CHECK-NOTES: :[[@LINE+10]]:18: note: function 'handler_multiple_calls' registered here as signal handler
  printf("1234");
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'printf' called here from 'handler_multiple_calls'
  // CHECK-NOTES: :[[@LINE+6]]:18: note: function 'handler_multiple_calls' registered here as signal handler
  f_extern();
  // first 'f_extern' call found only
}

void test_multiple_calls(void) {
  signal(SIGINT, handler_multiple_calls);
}

void f_recursive(void);

void handler_recursive(int) {
  f_recursive();
  printf("");
  // first 'printf' call (in other function) found only
}

void f_recursive(void) {
  f_extern();
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'f_extern' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'f_extern' called here from 'f_recursive'
  // CHECK-NOTES: :[[@LINE-9]]:3: note: function 'f_recursive' called here from 'handler_recursive'
  // CHECK-NOTES: :[[@LINE+10]]:18: note: function 'handler_recursive' registered here as signal handler
  printf("");
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'printf' called here from 'f_recursive'
  // CHECK-NOTES: :[[@LINE-14]]:3: note: function 'f_recursive' called here from 'handler_recursive'
  // CHECK-NOTES: :[[@LINE+5]]:18: note: function 'handler_recursive' registered here as signal handler
  handler_recursive(2);
}

void test_recursive(void) {
  signal(SIGINT, handler_recursive);
}

void f_multiple_paths(void) {
  printf("");
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-NOTES: :[[@LINE-2]]:3: note: function 'printf' called here from 'f_multiple_paths'
  // CHECK-NOTES: :[[@LINE+5]]:3: note: function 'f_multiple_paths' called here from 'handler_multiple_paths'
  // CHECK-NOTES: :[[@LINE+9]]:18: note: function 'handler_multiple_paths' registered here as signal handler
}

void handler_multiple_paths(int) {
  f_multiple_paths();
  f_multiple_paths();
}

void test_multiple_paths(void) {
  signal(SIGINT, handler_multiple_paths);
}

void handler_function_pointer(int) {
  void (*fp)(void) = f_extern;
  // Call with function pointer is not evalauted by the check.
  (*fp)();
}

void test_function_pointer(void) {
  signal(SIGINT, handler_function_pointer);
}

void test_other(void) {
  signal(SIGINT, handler_abort);
  signal(SIGINT, handler_signal);

  signal(SIGINT, _Exit);
  signal(SIGINT, other_call);
  // CHECK-NOTES: :[[@LINE-1]]:18: warning: 'other_call' may not be asynchronous-safe; using it as a signal handler may be dangerous [bugprone-signal-handler]
  signal(SIGINT, f_extern);
  // CHECK-NOTES: :[[@LINE-1]]:18: warning: 'f_extern' may not be asynchronous-safe; using it as a signal handler may be dangerous [bugprone-signal-handler]

  signal(SIGINT, SIG_IGN);
  signal(SIGINT, SIG_DFL);
}
