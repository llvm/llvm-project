// Check that sanitizer prints registers dump_registers on dump_registers=1
// RUN: %clangxx  %s -o %t
// RUN: %env_tool_opts=dump_registers=0 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-NODUMP --strict-whitespace
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-DUMP --strict-whitespace
//
// REQUIRES: x86_64-target-arch

#include <signal.h>

int main() {
  raise(SIGSEGV);
  // CHECK-DUMP: Register values
  // CHECK-DUMP-NEXT: rax = {{0x[0-9a-f]+}}  rbx = {{0x[0-9a-f]+}}  rcx = {{0x[0-9a-f]+}}  rdx = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT: rdi = {{0x[0-9a-f]+}}  rsi = {{0x[0-9a-f]+}}  rbp = {{0x[0-9a-f]+}}  rsp = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT:  r8 = {{0x[0-9a-f]+}}   r9 = {{0x[0-9a-f]+}}  r10 = {{0x[0-9a-f]+}}  r11 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT: r12 = {{0x[0-9a-f]+}}  r13 = {{0x[0-9a-f]+}}  r14 = {{0x[0-9a-f]+}}  r15 = {{0x[0-9a-f]+}}
  // CHECK-NODUMP-NOT: Register values
  return 0;
}
