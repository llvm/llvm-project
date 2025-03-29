// REQUIRES: riscv
// RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
// RUN: ld.lld %t.o -o %t.so -shared --noinhibit-exec 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump --no-print-imm-hex -s -d %t.so | FileCheck %s

// SEC: .got PROGBITS 0000000000002390

  .section .data
  .globl bar
bar:

  .globl _start
_start:  // PC = 0x33a8
// bar@GOTPCREL   = 0x2398 (got entry for `bar`) - 0x33a8 (.) = 0xf0efffff
// bar@GOTPCREL+4 = 0x2398 (got entry for `bar`) - 0x33ac (.) + 4 = 0xf0efffff
// bar@GOTPCREL-4 = 0x2398 (got entry for `bar`) - 0x33b0 (.) - 4 = 0xe4efffff
// CHECK:      Contents of section .data:
// CHECK-NEXT:  {{.*}} f0efffff f0efffff e4efffff
  .word bar@GOTPCREL
  .word bar@GOTPCREL+4
  .word bar@GOTPCREL-4

// WARN: relocation R_RISCV_GOT32_PCREL out of range: {{.*}} is not in [-2147483648, 2147483647]; references 'baz'
// WARN: relocation R_RISCV_GOT32_PCREL out of range: {{.*}} is not in [-2147483648, 2147483647]; references 'baz'
  .word baz@GOTPCREL+0xffffffff
  .word baz@GOTPCREL-0xffffffff
