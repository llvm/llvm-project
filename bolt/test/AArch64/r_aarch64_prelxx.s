// This test checks processing of R_AARCH64_PREL64/32/16 relocations
// S + A - P = Value
// S = P - A + Value

// mul(D1,0x100) == << 8
// mul(D2,0x10000) == << 16
// mul(D3,0x1000000) == << 24

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
// CHECKPREL32-NEXT: 04: [[#%x,D0:]] [[#%x,D1:]] [[#%x,D2:]] [[#%x,D3:]]

// 4 is offset in datatable
// 8 is addend
// CHECKPREL32: [[#DATATABLEADDR + 4 - 8 + D0 + mul(D1,0x100) + mul(D2,0x10000) + mul(D3,0x1000000)]] <_start>:

// RUN: llvm-objdump -D %t.bolt | FileCheck %s --check-prefix=CHECKPREL64
// CHECKPREL64: [[#%x,DATATABLEADDR:]] <datatable>:
// CHECKPREL64-NEXT: 00:
// CHECKPREL64-NEXT: 04:
// CHECKPREL64-NEXT: 08: [[#%x,D0:]] [[#%x,D1:]] [[#%x,D2:]] [[#%x,D3:]]
// CHECKPREL64-NEXT: 0c: 00 00 00 00

// 8 is offset in datatable
// 12 is addend
// CHECKPREL64: [[#DATATABLEADDR + 8 - 12 + D0 + mul(D1,0x100) + mul(D2,0x10000) + mul(D3,0x1000000)]] <_start>:

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
