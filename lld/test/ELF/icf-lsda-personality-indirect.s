# REQUIRES: x86
## Test that the CIE personality is compared through the DW_EH_PE_indirect
## pointer a compiler emits for it (a per-object "DW.ref" thunk): functions
## whose thunks resolve to the same personality fold, functions whose thunks
## resolve to different personalities do not.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --icf=all %t.o -o /dev/null --print-icf-sections | \
# RUN:   FileCheck %s --implicit-check-not=Z1av --implicit-check-not=Z1bv

## Z1av/Z1bv use thunks to different personalities: no fold.
## Z1cv/Z1dv use thunks to the same personality: fold.

# CHECK-DAG: selected section {{.*}}.o:(.text.Z1cv)
# CHECK-DAG: removing identical section {{.*}}.o:(.text.Z1dv)

.globl _Z1av, _Z1bv, _Z1cv, _Z1dv

.section .text.Z1av,"ax",@progbits
_Z1av:
  .cfi_startproc
  .cfi_personality 0x9b, thunk_a
  .cfi_lsda 27, .Llsda_a
  ret
  .cfi_endproc

.section .text.Z1bv,"ax",@progbits
_Z1bv:
  .cfi_startproc
  .cfi_personality 0x9b, thunk_b
  .cfi_lsda 27, .Llsda_b
  ret
  .cfi_endproc

.section .text.Z1cv,"ax",@progbits
_Z1cv:
  .cfi_startproc
  .cfi_personality 0x9b, thunk_c
  .cfi_lsda 27, .Llsda_c
  ret
  .cfi_endproc

.section .text.Z1dv,"ax",@progbits
_Z1dv:
  .cfi_startproc
  .cfi_personality 0x9b, thunk_d
  .cfi_lsda 27, .Llsda_d
  ret
  .cfi_endproc

.section .data.rel.ro.a,"aw",@progbits
thunk_a:
  .quad per1

.section .data.rel.ro.b,"aw",@progbits
thunk_b:
  .quad per2

.section .data.rel.ro.c,"aw",@progbits
thunk_c:
  .quad per1

.section .data.rel.ro.d,"aw",@progbits
thunk_d:
  .quad per1

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

.section .text.per1,"ax",@progbits
.globl per1
per1:
  ret

.section .text.per2,"ax",@progbits
.globl per2
per2:
  nop
  ret
