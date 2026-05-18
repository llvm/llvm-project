// Check that sanitizer prints registers dump_registers on dump_registers=1
// RUN: %clangxx  %s -o %t
// RUN: %env_tool_opts=dump_registers=0 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-NODUMP --strict-whitespace
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-DUMP --strict-whitespace
//
// REQUIRES: i386-target-arch

#include <signal.h>

int main() {
  raise(SIGSEGV);
  // CHECK-DUMP: Register values
  // CHECK-DUMP-NEXT: eax = {{0x[0-9a-f]+}}  ebx = {{0x[0-9a-f]+}}  ecx = {{0x[0-9a-f]+}}  edx = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT: edi = {{0x[0-9a-f]+}}  esi = {{0x[0-9a-f]+}}  ebp = {{0x[0-9a-f]+}}  esp = {{0x[0-9a-f]+}}
  // CHECK-NODUMP-NOT: Register values
  return 0;
}
