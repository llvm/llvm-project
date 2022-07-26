// This test checks processing of R_AARCH64_PREL64/32/16 relocations
// S + A - P = Value
// S = P - A + Value

// REQUIRES: system-linux

// RUN: %clang %cflags -nostartfiles -nostdlib %s -o %t.exe -mlittle-endian \
// RUN:     -Wl,-q -Wl,-z,max-page-size=4
// RUN: llvm-readelf -Wa %t.exe | FileCheck %s -check-prefix=CHECKPREL

// CHECKPREL:       R_AARCH64_PREL16      {{.*}} .dummy + 0
// CHECKPREL-NEXT:  R_AARCH64_PREL32      {{.*}} _start + 4
// CHECKPREL-NEXT:  R_AARCH64_PREL64      {{.*}} _start + 8

// RUN: llvm-bolt %t.exe -o %t.bolt
// RUN: llvm-objdump -D %t.bolt | FileCheck %s --check-prefix=CHECKPREL32

// CHECKPREL32: [[#%x,DATATABLEADDR:]] <datatable>:
// CHECKPREL32-NEXT: 00:
// CHECKPREL32-NEXT: 04: [[#%x,VALUE:]]

// 4 is offset in datatable
// 8 is addend
// CHECKPREL32: [[#DATATABLEADDR + 4 - 8 + VALUE]] <_start>:

// RUN: llvm-objdump -D %t.bolt | FileCheck %s --check-prefix=CHECKPREL64
// CHECKPREL64: [[#%x,DATATABLEADDR:]] <datatable>:
// CHECKPREL64-NEXT: 00:
// CHECKPREL64-NEXT: 04:
// CHECKPREL64-NEXT: 08: [[#%x,VALUE:]]
// CHECKPREL64-NEXT: 0c: 00000000

// 8 is offset in datatable
// 12 is addend
// CHECKPREL64: [[#DATATABLEADDR + 8 - 12 + VALUE]] <_start>:

  .section .text
  .align 4
  .globl _start
  .type _start, %function
_start:
  adr x0, datatable
  mov x0, #0
  ret 

.section .dummy, "da"
dummy:
  .word 0

  .data
  .align 8
datatable:
  .hword dummy - datatable
  .align 2
  .word _start - datatable
  .xword _start - datatable
