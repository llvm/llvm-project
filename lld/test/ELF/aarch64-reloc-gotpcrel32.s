// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld %t.o -o %t.so -shared --noinhibit-exec 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump --no-print-imm-hex -s -d %t.so | FileCheck %s

// SEC: .got PROGBITS 0000000000020390

  .section .data
  .globl bar
bar:

  .globl _start
_start:  // PC = 0x303a0
// bar@GOTPCREL   = 0x20390 (got entry for `bar`) - 0x303a0 (.) = 0xf0fffeff
// bar@GOTPCREL+4 = 0x20390 (got entry for `bar`) - 0x303a4 (.) + 4 = 0xf0fffeff
// bar@GOTPCREL-4 = 0x20390 (got entry for `bar`) - 0x303a8 (.) - 4 = 0xe4fffeff
// CHECK:      Contents of section .data:
// CHECK-NEXT:  {{.*}} f0fffeff f0fffeff e4fffeff
  .word bar@GOTPCREL
  .word bar@GOTPCREL+4
  .word bar@GOTPCREL-4

// WARN: relocation R_AARCH64_GOTPCREL32 out of range: {{.*}} is not in [-2147483648, 2147483647]; references 'baz'
// WARN: relocation R_AARCH64_GOTPCREL32 out of range: {{.*}} is not in [-2147483648, 2147483647]; references 'baz'
  .word baz@GOTPCREL+0xffffffff
  .word baz@GOTPCREL-0xffffffff
