// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared -soname=t2.so %t2.o -o %t2.so

// RUN: echo "SECTIONS { \
// RUN:   .text : { *(.text) } \
// RUN:   .plt : { *(.plt) } \
// RUN:   .got.plt : { *(.got.plt) } \
// RUN:   .dynstr : { *(.dynstr) } \
// RUN: }" > %t.script
// RUN: ld.lld %t1.o %t2.so -o %t.exe -z retpolineplt --script %t.script
// RUN: llvm-objdump --no-print-imm-hex -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: <.plt>:
// CHECK-NEXT: 10:       pushl   236
// CHECK-NEXT: 16:       pushl   %eax
// CHECK-NEXT: 17:       movl    240, %eax
// CHECK-NEXT: 1c:       calll   0x30 <.plt+0x20>
// CHECK-NEXT: 21:       pause
// CHECK-NEXT: 23:       lfence
// CHECK-NEXT: 26:       jmp     0x21 <.plt+0x11>
// CHECK-NEXT: 28:       int3
// CHECK-NEXT: 29:       int3
// CHECK-NEXT: 2a:       int3
// CHECK-NEXT: 2b:       int3
// CHECK-NEXT: 2c:       int3
// CHECK-NEXT: 2d:       int3
// CHECK-NEXT: 2e:       int3
// CHECK-NEXT: 2f:       int3
// CHECK-NEXT: 30:       movl    %ecx, (%esp)
// CHECK-NEXT: 33:       movl    4(%esp), %ecx
// CHECK-NEXT: 37:       movl    %eax, 4(%esp)
// CHECK-NEXT: 3b:       movl    %ecx, %eax
// CHECK-NEXT: 3d:       popl    %ecx
// CHECK-NEXT: 3e:       retl
// CHECK-NEXT: 3f:       int3
// CHECK-NEXT: 40:       pushl   %eax
// CHECK-NEXT: 41:       movl    244, %eax
// CHECK-NEXT: 46:       calll   0x30 <.plt+0x20>
// CHECK-NEXT: 4b:       jmp     0x21 <.plt+0x11>
// CHECK-NEXT: 50:       pushl   $0
// CHECK-NEXT: 55:       jmp     0x10 <.plt>
// CHECK-NEXT: 5a:       int3
// CHECK-NEXT: 5b:       int3
// CHECK-NEXT: 5c:       int3
// CHECK-NEXT: 5d:       int3
// CHECK-NEXT: 5e:       int3
// CHECK-NEXT: 5f:       int3
// CHECK-NEXT: 60:       pushl   %eax
// CHECK-NEXT: 61:       movl    248, %eax
// CHECK-NEXT: 66:       calll   0x30 <.plt+0x20>
// CHECK-NEXT: 6b:       jmp     0x21 <.plt+0x11>
// CHECK-NEXT: 70:       pushl   $8
// CHECK-NEXT: 75:       jmp     0x10 <.plt>
// CHECK-NEXT: 7a:       int3
// CHECK-NEXT: 7b:       int3
// CHECK-NEXT: 7c:       int3
// CHECK-NEXT: 7d:       int3
// CHECK-NEXT: 7e:       int3
// CHECK-NEXT: 7f:       int3

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
