// REQUIRES: riscv-registered-target
// RUN: %clang -target riscv64be-unknown-elf -### %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target riscv64be-unknown-elf -Wno-riscv-be-experimental -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOWARN

// CHECK: warning: big-endian RISC-V target support is experimental
// CHECK: "-triple" "riscv64be-unknown-unknown-elf"
// NOWARN-NOT: warning: big-endian RISC-V target support is experimental
// NOWARN: "-triple" "riscv64be-unknown-unknown-elf"

int foo(void) {
  return 0;
}
