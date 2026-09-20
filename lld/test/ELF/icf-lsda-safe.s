# REQUIRES: x86
## --icf=safe and LSDA-aware ICF:
## * an LSDA-bearing function that is not address-significant is foldable;
## * an address-significant (addrsig or exported) function stays unique even
##   when its LSDA and CIE are equivalent;
## * an object without an address-significance table stays conservative.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --icf=safe --print-icf-sections %t.o -o /dev/null | \
# RUN:   FileCheck %s --implicit-check-not=Z1cv --implicit-check-not=Z1dv \
# RUN:   --implicit-check-not=Z1ev --implicit-check-not=Z1fv
# RUN: ld.lld -shared --icf=safe --print-icf-sections %t.o -o /dev/null | \
# RUN:   FileCheck %s --check-prefix=EXPORT --implicit-check-not=Z1av --implicit-check-not=Z1bv
# RUN: llvm-objcopy --remove-section=.llvm_addrsig %t.o %t-noaddrsig.o
# RUN: ld.lld --icf=safe --print-icf-sections %t-noaddrsig.o -o /dev/null | \
# RUN:   FileCheck %s --check-prefix=NOADDRSIG --implicit-check-not=Z1av --implicit-check-not=Z1bv

# CHECK-DAG: selected section {{.*}}.o:(.text.Z1av)
# CHECK-DAG: removing identical section {{.*}}.o:(.text.Z1bv)
# EXPORT-DAG: selected section {{.*}}.o:(.gcc_except_table.a)
# EXPORT-DAG: removing identical section {{.*}}.o:(.gcc_except_table.b)
# NOADDRSIG-DAG: selected section {{.*}}.o:(.gcc_except_table.a)
# NOADDRSIG-DAG: removing identical section {{.*}}.o:(.gcc_except_table.b)

.globl _Z1av, _Z1bv, _Z1cv, _Z1dv, _Z1ev, _Z1fv

## Z1av/Z1bv: identical code and LSDA, not address-significant: fold.
.section .text.Z1av,"ax",@progbits
_Z1av:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_a
  ret
  .cfi_endproc
.section .text.Z1bv,"ax",@progbits
_Z1bv:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_b
  ret
  .cfi_endproc

## Z1cv/Z1dv: identical code and LSDA, but _Z1cv is address-significant.
.section .text.Z1cv,"ax",@progbits
_Z1cv:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_c
  ret
  .cfi_endproc
.section .text.Z1dv,"ax",@progbits
_Z1dv:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_d
  ret
  .cfi_endproc

## Z1ev/Z1fv: different LSDA: never fold.
.section .text.Z1ev,"ax",@progbits
_Z1ev:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_e
  ret
  .cfi_endproc
.section .text.Z1fv,"ax",@progbits
_Z1fv:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_f
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
.Llsda_e:
  .long 0x33333333
.section .gcc_except_table.f,"a",@progbits
.Llsda_f:
  .long 0x44444444

.addrsig
.addrsig_sym _Z1cv
