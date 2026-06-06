// Check that sanitizer prints registers dump_registers on dump_registers=1
// RUN: %clangxx  %s -o %t
// RUN: %env_tool_opts=dump_registers=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NODUMP
// RUN: not %run %t 2>&1 | FileCheck %s --strict-whitespace --check-prefix=CHECK-DUMP
//
// REQUIRES: aarch64-target-arch && glibc

#include <signal.h>

int main() {
  raise(SIGSEGV);
  // CHECK-DUMP: Register values
  // CHECK-DUMP-NEXT: x0 = {{0x[0-9a-f]+}}   x1 = {{0x[0-9a-f]+}}   x2 = {{0x[0-9a-f]+}}   x3 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT: x4 = {{0x[0-9a-f]+}}   x5 = {{0x[0-9a-f]+}}   x6 = {{0x[0-9a-f]+}}   x7 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT: x8 = {{0x[0-9a-f]+}}   x9 = {{0x[0-9a-f]+}}  x10 = {{0x[0-9a-f]+}}  x11 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT:x12 = {{0x[0-9a-f]+}}  x13 = {{0x[0-9a-f]+}}  x14 = {{0x[0-9a-f]+}}  x15 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT:x16 = {{0x[0-9a-f]+}}  x17 = {{0x[0-9a-f]+}}  x18 = {{0x[0-9a-f]+}}  x19 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT:x20 = {{0x[0-9a-f]+}}  x21 = {{0x[0-9a-f]+}}  x22 = {{0x[0-9a-f]+}}  x23 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT:x24 = {{0x[0-9a-f]+}}  x25 = {{0x[0-9a-f]+}}  x26 = {{0x[0-9a-f]+}}  x27 = {{0x[0-9a-f]+}}
  // CHECK-DUMP-NEXT:x28 = {{0x[0-9a-f]+}}   fp = {{0x[0-9a-f]+}}   lr = {{0x[0-9a-f]+}}   sp = {{0x[0-9a-f]+}}
  // CHECK-NODUMP-NOT: Register values
  return 0;
}
