// RUN: %clangxx -fsanitize=realtime -g %s -o %t
// RUN: %env_rtsan_opts=halt_on_error=false %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOSUPPRESSIONS
// RUN: %env_rtsan_opts=suppressions='%s.supp':halt_on_error=false:print_stats_on_exit=true %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Ensure that a suppression can name any function that the report
//         prints for an inlined call, wherever the call sits in the inlined
//         function.

#include <stdio.h>
#include <unistd.h>

volatile int Sink;

// The call is the last thing the inlined function does, so the return address
// already belongs to the caller.
__attribute__((always_inline)) static inline void SuppressedCallLast() {
  usleep(1);
}

void process() [[clang::nonblocking]] { SuppressedCallLast(); }

// The call is not the last thing the inlined function does, so the return
// address still belongs to it.
__attribute__((always_inline)) static inline void SuppressedCallThenWork() {
  usleep(1);
  Sink = Sink + 1;
}

void ProcessThenWork() [[clang::nonblocking]] { SuppressedCallThenWork(); }

// The suppression names the caller and not the inlined function.
__attribute__((always_inline)) static inline void InlinedHelper() { usleep(1); }

void SuppressedCaller() [[clang::nonblocking]] { InlinedHelper(); }

int main() {
  process();
  ProcessThenWork();
  SuppressedCaller();
  fprintf(stderr, "Done\n");
  return 0;
}

// CHECK-NOT: Intercepted call to real-time unsafe function
// CHECK: Done
// CHECK: Suppression count: 3

// CHECK-NOSUPPRESSIONS: Intercepted call to real-time unsafe function
// CHECK-NOSUPPRESSIONS: SuppressedCallLast
// CHECK-NOSUPPRESSIONS: Intercepted call to real-time unsafe function
// CHECK-NOSUPPRESSIONS: SuppressedCallThenWork
// CHECK-NOSUPPRESSIONS: Intercepted call to real-time unsafe function
// CHECK-NOSUPPRESSIONS: InlinedHelper
// CHECK-NOSUPPRESSIONS: SuppressedCaller
// CHECK-NOSUPPRESSIONS: Done
