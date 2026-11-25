// RUN: llvm-mc -triple riscv64 -mattr=+d,+c -target-abi=lp64d -filetype=obj -o %t.lp64d.o %s
// RUN: ld.lld -q -o %t.lp64d %t.lp64d.o
// RUN: llvm-bolt -o %t.lp64d.bolt %t.lp64d
// RUN: llvm-readelf -h %t.lp64d.bolt | FileCheck --check-prefix=LP64D %s

// RUN: llvm-mc -triple riscv64 -mattr=+f -target-abi=lp64f -filetype=obj -o %t.lp64f.o %s
// RUN: ld.lld -q -o %t.lp64f %t.lp64f.o
// RUN: llvm-bolt -o %t.lp64f.bolt %t.lp64f
// RUN: llvm-readelf -h %t.lp64f.bolt | FileCheck --check-prefix=LP64F %s

// RUN: llvm-mc -triple riscv64 -mattr=+e -target-abi=lp64e -filetype=obj -o %t.lp64e.o %s
// RUN: ld.lld -q -o %t.lp64e %t.lp64e.o
// RUN: llvm-bolt -o %t.lp64e.bolt %t.lp64e
// RUN: llvm-readelf -h %t.lp64e.bolt | FileCheck --check-prefix=LP64E %s

// RUN: llvm-mc -triple riscv32 -mattr=+d -target-abi=ilp32d -filetype=obj -o %t.ilp32d.o %s
// RUN: ld.lld -q -o %t.ilp32d %t.ilp32d.o
// RUN: llvm-bolt -o %t.ilp32d.bolt %t.ilp32d
// RUN: llvm-readelf -h %t.ilp32d.bolt | FileCheck --check-prefix=ILP32D %s

// LP64D:  Flags: 0x5, RVC, double-float ABI
// LP64F:  Flags: 0x2, single-float ABI
// LP64E:  Flags: 0x8, RVE
// ILP32D: Flags: 0x4, double-float ABI

  .text
  .globl _start
  .type _start, @function
_start:
  call foo
  ret
  .size _start, .-_start

  .globl foo
  .type foo, @function
foo:
  ret
  .size foo, .-foo
