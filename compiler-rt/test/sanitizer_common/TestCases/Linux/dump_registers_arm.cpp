// Check that sanitizer prints registers dump_registers on dump_registers=1
// RUN: %clangxx  %s -o %t
// RUN: %env_tool_opts=dump_registers=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NODUMP
// RUN: not %run %t 2>&1 | FileCheck %s --strict-whitespace --check-prefix=CHECK-DUMP
//
// REQUIRES: arm-target-arch && glibc

#include <signal.h>

int main() {
  raise(SIGSEGV);
  // CHECK-DUMP: Register values
  // CHECK-DUMP-NEXT: r0 = {{0x[0-9a-f]+}}   r1 = {{0x[0-9a-f]+}}   r2 = {{0x[0-9a-f]+}}   r3 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT: r4 = {{0x[0-9a-f]+}}   r5 = {{0x[0-9a-f]+}}   r6 = {{0x[0-9a-f]+}}   r7 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT: r8 = {{0x[0-9a-f]+}}   r9 = {{0x[0-9a-f]+}}  r10 = {{0x[0-9a-f]+}}  r11 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT:r12 = {{0x[0-9a-f]+}}   sp = {{0x[0-9a-f]+}}   lr = {{0x[0-9a-f]+}}   pc = {{0x[0-9a-f]+}}
  // CHECK-NODUMP-NOT: Register values
  return 0;
}
