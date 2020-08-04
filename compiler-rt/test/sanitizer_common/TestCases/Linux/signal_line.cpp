// Test line numbers in signal handlers
// Fails with debug checks: https://bugs.llvm.org/show_bug.cgi?id=46860
// XFAIL: !compiler-rt-optimized && tsan

// RUN: %clangxx %s -o %t -O0
// RUN: %env_tool_opts=handle_segv=1:print_stacktrace=1 not %run %t 1 2>&1 | FileCheck --check-prefixes=CHECK1,CHECK %s
// RUN: %env_tool_opts=handle_segv=1:print_stacktrace=1 not %run %t 2 2>&1 | FileCheck --check-prefixes=CHECK2,CHECK %s

#include <cstdio>
#include <string>

// CHECK: [[SAN:.*Sanitizer]]:DEADLYSIGNAL
// CHECK: ERROR: [[SAN]]: SEGV on unknown address {{0x[^ ]*}} (pc
int main(int argc, char **argv) {
  int n = atoi(argv[1]);

  if (n == 1)
    *((volatile int *)0x0) = __LINE__;
  // CHECK1: #{{[0-9]+ .*}}main {{.*}}signal_line.cpp:[[@LINE-1]]:[[TAB:[0-9]+]]
  // CHECK1: SUMMARY: [[SAN]]: SEGV {{.*}}signal_line.cpp:[[@LINE-2]]:[[TAB]] in main

  if (n == 2)
    *((volatile int *)0x1) = __LINE__;
  // CHECK2: #{{[0-9]+ .*}}main {{.*}}signal_line.cpp:[[@LINE-1]]:[[TAB:[0-9]+]]
  // CHECK2: SUMMARY: [[SAN]]: SEGV {{.*}}signal_line.cpp:[[@LINE-2]]:[[TAB]] in main
}
