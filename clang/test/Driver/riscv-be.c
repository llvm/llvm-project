// REQUIRES: riscv-registered-target
// RUN: %clang -target riscv64be-unknown-elf -### %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: warning: big-endian RISC-V target support is experimental
// CHECK: "-triple" "riscv64be-unknown-unknown-elf"

int foo(void) {
  return 0;
}
