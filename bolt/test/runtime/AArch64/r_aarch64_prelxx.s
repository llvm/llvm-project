// This test checks processing of R_AARCH64_PREL64/32/16 relocations

// RUN: %clang %cflags -nostartfiles -nostdlib %s -o %t.exe -Wl,-q \
// RUN:     -Wl,-z,max-page-size=4
// RUN: llvm-readelf -Wa %t.exe | FileCheck %s -check-prefix=CHECKPREL

// CHECKPREL:       R_AARCH64_PREL16      {{.*}} .dummy + 0
// CHECKPREL-NEXT:  R_AARCH64_PREL32      {{.*}} _start + 4
// CHECKPREL-NEXT:  R_AARCH64_PREL64      {{.*}} _start + 8

// RUN: llvm-bolt %t.exe -o %t.bolt
// RUN: llvm-readobj -S --section-data %t.bolt | FileCheck %s

// CHECK: Name: .data
// CHECK: SectionData (
// CHECK:   0000: FCFF0000 44FF3F00 44FF3F00 00000000
// CHECK: )

  .text
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

.section .data
datatable:
  .hword dummy - datatable
  .align 2
  .word _start - datatable
  .xword _start - datatable
