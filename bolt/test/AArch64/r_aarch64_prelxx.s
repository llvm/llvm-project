// This test checks processing of R_AARCH64_PREL64/32/16 relocations
// S + A - P = Value
// S = P - A + Value

// REQUIRES: system-linux

// RUN: %clang %cflags -nostartfiles -nostdlib %s -o %t.exe -mlittle-endian \
// RUN:     -Wl,-q -Wl,-z,max-page-size=4 -Wl,--no-relax
// RUN: llvm-readelf -Wa %t.exe | FileCheck %s -check-prefix=CHECKPREL

// CHECKPREL:       R_AARCH64_PREL16      {{.*}} .dummy + 0
// CHECKPREL-NEXT:  R_AARCH64_PREL32      {{.*}} _start + 4
// CHECKPREL-NEXT:  R_AARCH64_PREL64      {{.*}} _start + 8

// RUN: llvm-bolt %t.exe -o %t.bolt
// RUN: llvm-objdump -D %t.bolt | FileCheck %s --check-prefix=CHECKPREL32

// CHECKPREL32: [[#%x,DATATABLEADDR:]] <datatable>:
// CHECKPREL32-NEXT: 00:
// CHECKPREL32-NEXT: 04: {{.*}} .word 0x[[#%x,VALUE:]]

// CHECKPREL32: [[#DATATABLEADDR + VALUE]] <_start>:

// RUN: llvm-objdump -D %t.bolt | FileCheck %s --check-prefix=CHECKPREL64
// CHECKPREL64: [[#%x,DATATABLEADDR:]] <datatable>:
// CHECKPREL64-NEXT: 00:
// CHECKPREL64-NEXT: 04:
// CHECKPREL64-NEXT: 08: {{.*}} .word 0x[[#%x,VALUE:]]
// CHECKPREL64-NEXT: 0c: {{.*}} .word 0x00000000

// CHECKPREL64: [[#DATATABLEADDR + VALUE]] <_start>:

  .section .text
  .align 4
  .globl _start
  .type _start, %function
_start:
  adrp x0, datatable
  add x0, x0, :lo12:datatable
  mov x0, #0
  ret

.section .dummy, "a", @progbits
dummy:
  .word 0

  .data
  .align 8
datatable:
  .hword dummy - datatable
  .align 2
  .word _start - datatable
  .xword _start - datatable
