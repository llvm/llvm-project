// RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.o %s
// RUN: ld.lld -q -o %t %t.o
// RUN: llvm-bolt -o %t.bolt %t
// RUN: llvm-readelf -s %t.bolt | FileCheck --check-prefix=SYM %s
// RUN: llvm-readelf -x .data %t.bolt | FileCheck --check-prefix=DATA %s

// SYM: {{0+}}400000 {{.*}} _start{{$}}

// DATA: Hex dump of section '.data':
// DATA-NEXT: 00004000 00000000

  .data
  .globl d
  .p2align 3
d:
  .dword _start

  .text
  .globl _start
  .p2align 1
_start:
  ret
  ## Dummy relocation to force relocation mode; without it, _start will not be
  ## moved to a new address.
  .reloc 0, R_RISCV_NONE
  .size _start, .-_start
