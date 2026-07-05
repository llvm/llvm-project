# REQUIRES: ppc
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=ppc64le %t/asm -o %t.o
# RUN: not ld.lld --script %t/lds %t.o -o %t1 2>&1 | FileCheck %s
# RUN: ld.lld --script %t/lds %t.o -o %t1 --noinhibit-exec
# RUN: rm %t.o %t1
# RUN: llvm-mc -filetype=obj -triple=ppc64le -defsym HIDDEN=1 %t/asm -o %t.o
# RUN: not ld.lld -shared --script %t/lds %t.o -o %t1 2>&1 | FileCheck %s
# RUN: ld.lld -shared --script %t/lds %t.o -o %t1 --noinhibit-exec
# RUN: rm %t.o %t1

# RUN: llvm-mc -filetype=obj -triple=ppc64 %t/asm -o %t.o
# RUN: not ld.lld --script %t/lds %t.o -o %t1 2>&1 | FileCheck %s
# RUN: ld.lld --script %t/lds %t.o -o %t1 --noinhibit-exec
# RUN: rm %t.o %t1
# RUN: llvm-mc -filetype=obj -triple=ppc64 -defsym HIDDEN=1 %t/asm -o %t.o
# RUN: not ld.lld -shared --script %t/lds %t.o -o %t1 2>&1 | FileCheck %s
# RUN: ld.lld -shared --script %t/lds %t.o -o %t1 --noinhibit-exec
# RUN: rm %t.o %t1

## Note: GNU ld uses sldi 11, 11, 34 and add 12, 11, 12 to support offsets beyond +-2**33.
# CHECK: error: R12 setup stub offset is out of range: 8589934592 is not in [-8589934592, 8589934591]; references 'high'
# CHECK-NEXT: >>> defined in {{.*}}.o

//--- asm
.section .text_low, "ax", %progbits
.globl _start
_start:
  bl high@notoc
  blr

.section .text_high, "ax", %progbits
.ifdef HIDDEN
.hidden high
.endif
.globl high
high:
  blr

//--- lds
PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .text_low 0x2000 : { *(.text_low) } :low
  .text_high 0x200002010 : { *(.text_high) } :high
}
