# REQUIRES: x86
## Test that LSDA-aware ICF also compares the CIE (personality, encodings, CFI):
## * identical code + LSDA but different CIE personalities are not folded;
## * identical code + LSDA + same personality are folded;
## * --keep-unique on an LSDA symbol keeps the associated functions unique.

## Test REL.
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t1.o
# RUN: ld.lld --icf=all %t1.o -o /dev/null --print-icf-sections | \
# RUN:   FileCheck %s --implicit-check-not=Z1av --implicit-check-not=Z1bv
## Test RELA.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t2.o
# RUN: ld.lld --icf=all %t2.o -o /dev/null --print-icf-sections | \
# RUN:   FileCheck %s --implicit-check-not=Z1av --implicit-check-not=Z1bv
## --keep-unique on an exported LSDA symbol prevents folding of its functions.
# RUN: ld.lld --icf=all --keep-unique=lsdae %t2.o -o /dev/null --print-icf-sections | \
# RUN:   FileCheck %s --check-prefix=KEEPUNIQUE --implicit-check-not=Z1ev --implicit-check-not=Z1fv

# CHECK-DAG: selected section {{.*}}.o:(.text.Z1cv)
# CHECK-DAG: removing identical section {{.*}}.o:(.text.Z1dv)
## Z1ev/Z1fv have an exported LSDA symbol; ICF may still fold them because the
## LSDA and CIE are equivalent.
# CHECK-DAG: selected section {{.*}}.o:(.text.Z1ev)
# CHECK-DAG: removing identical section {{.*}}.o:(.text.Z1fv)

# KEEPUNIQUE-DAG: selected section {{.*}}.o:(.text.Z1cv)
# KEEPUNIQUE-DAG: removing identical section {{.*}}.o:(.text.Z1dv)

.globl _Z1av, _Z1bv, _Z1cv, _Z1dv, _Z1ev, _Z1fv

## Z1av and Z1bv have identical code and identical LSDA, but different CIE
## personalities, so they must not be folded.
.section .text.Z1av,"ax",@progbits
_Z1av:
  .cfi_startproc
  .cfi_personality 27, per1
  .cfi_lsda 27, .Llsda_a
  ret
  .cfi_endproc

.section .text.Z1bv,"ax",@progbits
_Z1bv:
  .cfi_startproc
  .cfi_personality 27, per2
  .cfi_lsda 27, .Llsda_b
  ret
  .cfi_endproc

## Z1cv and Z1dv share the personality and have identical LSDA: fold.
.section .text.Z1cv,"ax",@progbits
_Z1cv:
  .cfi_startproc
  .cfi_personality 27, per1
  .cfi_lsda 27, .Llsda_c
  ret
  .cfi_endproc

.section .text.Z1dv,"ax",@progbits
_Z1dv:
  .cfi_startproc
  .cfi_personality 27, per1
  .cfi_lsda 27, .Llsda_d
  ret
  .cfi_endproc

## Z1ev and Z1fv are folded unless --keep-unique keeps their LSDA unique.
.section .text.Z1ev,"ax",@progbits
_Z1ev:
  .cfi_startproc
  .cfi_personality 27, per1
  .cfi_lsda 27, lsdae
  ret
  .cfi_endproc

.section .text.Z1fv,"ax",@progbits
_Z1fv:
  .cfi_startproc
  .cfi_personality 27, per1
  .cfi_lsda 27, lsdaef
  ret
  .cfi_endproc

.section .gcc_except_table.a,"a",@progbits
.Llsda_a:
  .long 0x11111111

.section .gcc_except_table.b,"a",@progbits
.Llsda_b:
  .long 0x11111111

.section .gcc_except_table.c,"a",@progbits
.Llsda_c:
  .long 0x22222222

.section .gcc_except_table.d,"a",@progbits
.Llsda_d:
  .long 0x22222222

.section .gcc_except_table.e,"a",@progbits
.globl lsdae
lsdae:
  .long 0x33333333

.section .gcc_except_table.f,"a",@progbits
.globl lsdaef
lsdaef:
  .long 0x33333333

## Personalities referenced by the CIEs above. They are deliberately
## different (different contents) so that they are not folded into each other.
.section .text.per1,"ax",@progbits
.globl per1
per1:
  ret

.section .text.per2,"ax",@progbits
.globl per2
per2:
  nop
  ret
