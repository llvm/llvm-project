# REQUIRES: x86
## Test that the CIE personality of functions in different object files is
## compared through each object's DW_EH_PE_indirect thunk: thunks resolving to
## the same personality fold, a thunk resolving to a different personality
## prevents folding.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/c.s -o %t/c.o
# RUN: ld.lld --icf=all --print-icf-sections %t/a.o %t/b.o -o /dev/null | \
# RUN:   FileCheck %s --check-prefix=FOLD
# RUN: ld.lld --icf=all --print-icf-sections %t/a.o %t/c.o -o /dev/null | \
# RUN:   FileCheck %s --check-prefix=NOFOLD --implicit-check-not=Z1av --implicit-check-not=Z1cv

# FOLD-DAG: selected section {{.*}}a.o:(.text.Z1av)
# FOLD-DAG: removing identical section {{.*}}b.o:(.text.Z1bv)
## The equivalent exception tables still fold even though the functions do not.
# NOFOLD-DAG: selected section {{.*}}a.o:(.gcc_except_table.a)
# NOFOLD-DAG: removing identical section {{.*}}c.o:(.gcc_except_table.c)

#--- a.s
.globl _Z1av, __gxx_personality_v0
.section .text.Z1av,"ax",@progbits
_Z1av:
  .cfi_startproc
  .cfi_personality 0x9b, thunk_a
  .cfi_lsda 27, .Llsda_a
  ret
  .cfi_endproc
.section .data.rel.ro.a,"aw",@progbits
thunk_a:
  .quad __gxx_personality_v0
.section .gcc_except_table.a,"a",@progbits
.Llsda_a:
  .long 0x11111111
.section .text.per,"ax",@progbits
__gxx_personality_v0:
  ret

#--- b.s
.globl _Z1bv
.section .text.Z1bv,"ax",@progbits
_Z1bv:
  .cfi_startproc
  .cfi_personality 0x9b, thunk_b
  .cfi_lsda 27, .Llsda_b
  ret
  .cfi_endproc
.section .data.rel.ro.b,"aw",@progbits
thunk_b:
  .quad __gxx_personality_v0
.section .gcc_except_table.b,"a",@progbits
.Llsda_b:
  .long 0x11111111

#--- c.s
.globl _Z1cv
.section .text.Z1cv,"ax",@progbits
_Z1cv:
  .cfi_startproc
  .cfi_personality 0x9b, thunk_c
  .cfi_lsda 27, .Llsda_c
  ret
  .cfi_endproc
.section .data.rel.ro.c,"aw",@progbits
thunk_c:
  .quad other_personality
.section .gcc_except_table.c,"a",@progbits
.Llsda_c:
  .long 0x11111111
.section .text.per2,"ax",@progbits
other_personality:
  nop
  ret
