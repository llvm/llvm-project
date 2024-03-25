# REQUIRES: ppc
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=ppc64le %t/asm -o %t.o
# RUN: not ld.lld --script %t/lds %t.o -o %t1 2>&1 | FileCheck %s
# RUN: ld.lld --script %t/lds %t.o -o %t1 --noinhibit-exec
# RUN: rm %t.o %t1

# RUN: llvm-mc -filetype=obj -triple=ppc64 %t/asm -o %t.o
# RUN: not ld.lld --script %t/lds %t.o -o %t1 2>&1 | FileCheck %s
# RUN: ld.lld --script %t/lds %t.o -o %t1 --noinhibit-exec
# RUN: rm %t.o %t1

# CHECK: error: R12 setup stub offset is out of range: 8589934592 is not in [-8589934592, 8589934591]; references 'callee'
# CHECK-NEXT: >>> defined in {{.*}}.o

//--- asm
.section .text_high, "ax", %progbits
callee:
  .Lfunc_gep1:
  addis 2, 12, .TOC.-.Lfunc_gep1@ha
  addi 2, 2, .TOC.-.Lfunc_gep1@l
  .Lfunc_lep1:
  .localentry callee, .Lfunc_lep1-.Lfunc_gep1
  addis 4, 2, global@toc@ha
  lwz 4, global@toc@l(4)
  blr

.section .text_low, "ax", %progbits
caller:
  .localentry caller, 1
  bl callee@notoc
  blr
global:
  .long 0

//--- lds
PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .text_low 0x2000 : { *(.text_low) } :low
  .text_high 0x200002010 : { *(.text_high) } :high
}
