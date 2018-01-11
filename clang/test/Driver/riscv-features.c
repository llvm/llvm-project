// RUN: %clang -target riscv32-unknown-elf -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -### %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: fno-signed-char
