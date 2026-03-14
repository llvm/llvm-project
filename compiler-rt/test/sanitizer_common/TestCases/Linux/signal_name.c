// RUN: %clang -O1 %s -o %t
// RUN: %env_tool_opts=handle_sigfpe=2 not %run %t 0 2>&1 | FileCheck %s -DSIGNAME=FPE
// RUN: %env_tool_opts=handle_sigill=2 not %run %t 1 2>&1 | FileCheck %s -DSIGNAME=ILL
// RUN: %env_tool_opts=handle_abort=2 not %run %t 2 2>&1 | FileCheck %s -DSIGNAME=ABRT
// RUN: %env_tool_opts=handle_segv=2 not %run %t 3 2>&1 | FileCheck %s -DSIGNAME=SEGV
// RUN: %env_tool_opts=handle_sigbus=2 not %run %t 4 2>&1 | FileCheck %s -DSIGNAME=BUS
// RUN: %env_tool_opts=handle_sigtrap=2 not %run %t 5 2>&1 | FileCheck %s -DSIGNAME=TRAP

// FIXME: Hwasan misclassify TRAP as tag missmatch.
// XFAIL: hwasan && !hwasan-aliasing

#include <signal.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 2)
    return 0;
  int signals[] = {SIGFPE, SIGILL, SIGABRT, SIGSEGV, SIGBUS, SIGTRAP};
  raise(signals[atoi(argv[1])]);
}

// CHECK: Sanitizer:DEADLYSIGNAL
// CHECK: Sanitizer: [[SIGNAME]] on unknown address
