// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: echo 'SECTIONS { \
// RUN:   . = 0x10000; \
// RUN:   .text : { *(.text) } \
// RUN:   .got : { PROVIDE_HIDDEN(__pic_static_base = .); PROVIDE_HIDDEN(_GLOBAL_OFFSET_TABLE_ = .); *(.got.plt) *(.got) } \
// RUN:   .data : { *(.data) } \
// RUN: }' > %t.script
// RUN: ld.lld --pie --script %t.script %t.o -o %t
// RUN: llvm-readelf -s -r -x .got %t | FileCheck %s --check-prefix=READELF \
// RUN:   --implicit-check-not=R_ARM_ABS32 --implicit-check-not=R_ARM_REL32
// RUN: llvm-objdump --triple=armv7a-none-linux-gnueabi -d --no-show-raw-insn %t | \
// RUN:   FileCheck %s --check-prefix=DISASM

// This is a minimal static-base PIC startup pattern. The first few
// instructions initialize r9 without using the GOT, because GOT-based data
// access itself depends on r9 already being valid. Later code can then use
// R_ARM_GOT_BREL offsets relative to that base.

.syntax unified
.text
.globl _start
.p2align 2
_start:
  adr r9, .Lstatic_base_delta
  ldr r12, [r9]
  add r9, r9, r12

  ldr r0, .Ldata_got
  ldr r0, [r9, r0]
  ldr r0, [r0]
  bx lr

.p2align 2
.Lstatic_base_delta:
  .word __pic_static_base - .
.Ldata_got:
  .word data_object(GOT)

.data
.globl data_object
.p2align 2
data_object:
  .word 0x12345678

// READELF:      Relocation section '.rel.dyn'
// READELF-NEXT:  Offset
// READELF-NEXT: 00010068 {{.*}} R_ARM_RELATIVE

// READELF:      {{[0-9]+}}: 00010068 {{.*}} LOCAL  HIDDEN  {{.*}} __pic_static_base
// READELF:      {{[0-9]+}}: 000100d4 {{.*}} GLOBAL DEFAULT {{.*}} data_object

// READELF:      Hex dump of section '.got':
// READELF-NEXT: 0x00010068 d4000100

// DISASM:      <_start>:
// DISASM-NEXT:   10044:       add     r9, pc, #20
// DISASM-NEXT:   10048:       ldr     r12, [r9]
// DISASM-NEXT:   1004c:       add     r9, r9, r12
// DISASM-NEXT:   10050:       ldr     r0, [pc, #0xc]
// DISASM-NEXT:   10054:       ldr     r0, [r9, r0]
// DISASM-NEXT:   10058:       ldr     r0, [r0]
// DISASM-NEXT:   1005c:       bx      lr
// DISASM-NEXT:   10060:       08 00 00 00
// DISASM-NEXT:   10064:       00 00 00 00
