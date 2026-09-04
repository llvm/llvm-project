# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o

## Every branch form reaches the same nearby target.
# RUN: ld.lld %t.o --defsym=w16=_start+0x18 --defsym=w19=_start+0x18 \
# RUN:   --defsym=w22=_start+0x18 --defsym=w30=_start+0x18 \
# RUN:   --defsym=far=_start+0x100000 -o %t
# RUN: llvm-objdump -d -s -j .text --no-print-imm-hex %t | FileCheck %s

## Displacements at the top of each field. R_SPARC_WDISP16 splits its
## displacement between bits 21:20 and 13:0, so d16hi is only exercised here.
# RUN: ld.lld %t.o --defsym=w16=_start+0x1fffc --defsym=w19=_start+0x100000 \
# RUN:   --defsym=w22=_start+0x800004 --defsym=w30=_start+0x80000008 \
# RUN:   --defsym=far=_start+0x100000 -o %t.limits
# RUN: llvm-objdump -d %t.limits | FileCheck --check-prefix=LIMITS %s

## R_SPARC_DISP8, R_SPARC_DISP16, R_SPARC_DISP32 and R_SPARC_DISP64 close out
## .text, each holding target - . = -4, -5, -7 and -11. The 16-, 32- and 64-bit
## stores are unaligned; SPARC has no unaligned variant of the DISP relocations.
# CHECK:      Contents of section .text:
# CHECK:      fcfffbff
# CHECK-NEXT: fffff9ff ffffffff fffff5

## far - . is 0xffff0 for the sethi and 0xfffec for the or.
# CHECK:      <_start>:
# CHECK-NEXT:   0a c8 40 06 brnz %g1, 0x[[#%x,TGT:]]
# CHECK-NEXT:   02 48 00 05 be %icc, 0x[[#TGT]]
# CHECK-NEXT:   10 80 00 04 ba 0x[[#TGT]]
# CHECK-NEXT:   40 00 00 03 call 0x[[#TGT]]
# CHECK-NEXT:   05 00 03 ff sethi 1023, %g2
# CHECK-NEXT:   84 10 a3 ec or %g2, 1004, %g2

## w16 - . is 0x1fffc, so d16hi is 0b01 and d16lo is 0x3fff.
## w19 - . is 0xffffc, w22 - . is 0x7ffffc and w30 - . is 0x7ffffffc.
# LIMITS:      <_start>:
# LIMITS-NEXT:   0a d8 7f ff brnz
# LIMITS-NEXT:   02 4b ff ff be
# LIMITS-NEXT:   10 9f ff ff ba
# LIMITS-NEXT:   5f ff ff ff call

.section .text.a,"ax",@progbits
.globl _start
_start:
## R_SPARC_WDISP16, R_SPARC_WDISP19, R_SPARC_WDISP22, R_SPARC_WDISP30
  brnz    %g1, w16
  be,pt   %icc, w19
  ba      w22
  call    w30
## R_SPARC_PC22, R_SPARC_PC10
  sethi   %pc22(far), %g2
  or      %g2, %pc10(far), %g2

.section .text.b,"ax",@progbits
target:
  nop

.section .text.c,"ax",@progbits
  .byte  target - .
  .half  target - .
  .word  target - .
  .xword target - .
