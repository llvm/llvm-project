// This test checks that BOLT disassembles PLT stubs in binaries using BTI,
// while keeping them not disassembled in non-BTI binaries.

// RUN: %clang -fuse-ld=lld --target=aarch64-unknown-linux-gnu %s -o %t.exe \
// RUN: -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm | FileCheck %s

// RUN: %clang -fuse-ld=lld --target=aarch64-unknown-linux-gnu \
// RUN: -mbranch-protection=standard %s -o %t.bti.exe -Wl,-q -Wl,-z,force-bti
// RUN: llvm-bolt %t.bti.exe -o %t.bolt --print-disasm | FileCheck %s \
// RUN: --check-prefix=CHECK-BTI

// For the non-BTI binary, PLTs should not be disassembled.
// CHECK-NOT: Binary Function "{{.*}}@PLT" after disassembly {

// Check that PLTs are disassembled for the BTI binary.
// CHECK-BTI: Binary Function "__libc_start_main@PLT" after disassembly {
// CHECK-BTI: adrp
// CHECK-BTI-NEXT: ldr
// CHECK-BTI-NEXT: add
// CHECK-BTI-NEXT: br
// CHECK-BTI: End of Function "__libc_start_main@PLT"

#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  if (argc > 3)
    exit(42);
  else
    printf("Number of args: %d\n", argc);
}
