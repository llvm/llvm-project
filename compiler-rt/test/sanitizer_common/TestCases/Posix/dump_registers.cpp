// Check that sanitizer prints registers dump_registers on dump_registers=1
// RUN: %clangxx  %s -o %t
// RUN: %env_tool_opts=dump_registers=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NODUMP
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-DUMP
//
// FIXME: Implement.
// XFAIL: *

#include <signal.h>

int main() {
  raise(SIGSEGV);
  // CHECK-DUMP: Register values
  // CHECK-NODUMP-NOT: Register values
  return 0;
}
