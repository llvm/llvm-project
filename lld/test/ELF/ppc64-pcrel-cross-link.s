# REQUIRES: ppc
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/callees.s -o %t-callees.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/callertoc.s -o %t-callertoc.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/callernotoc.s -o %t-callernotoc.o
# RUN: ld.lld -T %t/lds %t-callees.o %t-callernotoc.o %t-callertoc.o -o %t-r12setup
# RUN: ld.lld -T %t/ldsswap %t-callees.o %t-callernotoc.o %t-callertoc.o -o %t-r2save

# RUN: llvm-objdump -d --no-show-raw-insn %t-r12setup | \
# RUN:   FileCheck %s --check-prefix=NOSWAP
# RUN: llvm-objdump -d --no-show-raw-insn %t-r2save | \
# RUN:   FileCheck %s --check-prefix=SWAP

## This test checks that it is possible to mix TOC and NOTOC functions and have
## the correct thunks for all of the calls.

# NOSWAP-LABEL: 01001000 <func>:
# NOSWAP-NEXT:    blr
# NOSWAP-LABEL: 01002000 <callee_toc>:
# NOSWAP:         bl 0x1001000
# NOSWAP:         blr
# NOSWAP-LABEL: 01002018 <callee_notoc>:
# NOSWAP:         bl 0x1001000
# NOSWAP:         blr
# NOSWAP-LABEL: 10030000 <caller_notoc>:
# NOSWAP-NEXT:    bl 0x10030010
# NOSWAP-NEXT:    bl 0x10030030
# NOSWAP-NEXT:    blr
# NOSWAP-LABEL: 10030010 <__gep_setup_callee_toc>:
# NOSWAP:         bctr
# NOSWAP-LABEL: 10030030 <__gep_setup_callee_notoc>:
# NOSWAP:         bctr
# NOSWAP-LABEL: 10040000 <caller_toc>:
# NOSWAP:         bl 0x10040020
# NOSWAP-NEXT:    nop
# NOSWAP-NEXT:    bl 0x10040040
# NOSWAP-NEXT:    ld 2, 24(1)
# NOSWAP-NEXT:    blr
# NOSWAP-LABEL: 10040020 <__long_branch_callee_toc>:
# NOSWAP:         bctr
# NOSWAP-LABEL: 10040040 <__toc_save_callee_notoc>:
# NOSWAP-NEXT:    std 2, 24(1)
# NOSWAP:         bctr

# SWAP-LABEL: 01001000 <func>:
# SWAP-NEXT:    blr
# SWAP-LABEL: 01002000 <callee_toc>:
# SWAP:         bl 0x1001000
# SWAP:         blr
# SWAP-LABEL: 01002018 <callee_notoc>:
# SWAP:         bl 0x1001000
# SWAP:         blr
# SWAP-LABEL: 10030000 <caller_toc>:
# SWAP:         bl 0x10030020
# SWAP-NEXT:    nop
# SWAP-NEXT:    bl 0x10030040
# SWAP-NEXT:    ld 2, 24(1)
# SWAP-NEXT:    blr
# SWAP-LABEL: 10030020 <__long_branch_callee_toc>:
# SWAP:         bctr
# SWAP-LABEL: 10030040 <__toc_save_callee_notoc>:
# SWAP-NEXT:    std 2, 24(1)
# SWAP:         bctr
# SWAP-LABEL: 10040000 <caller_notoc>:
# SWAP-NEXT:    bl 0x10040010
# SWAP-NEXT:    bl 0x10040030
# SWAP-NEXT:    blr
# SWAP-LABEL: 10040010 <__gep_setup_callee_toc>:
# SWAP:         bctr
# SWAP-LABEL: 10040030 <__gep_setup_callee_notoc>:
# SWAP:         bctr

#--- lds
SECTIONS {
  .text_func         0x1001000 : { *(.text_func) }
  .text_callee       0x1002000 : { *(.text_callee) }
  .text_caller_notoc 0x10030000 : { *(.text_caller_notoc) }
  .text_caller_toc   0x10040000 : { *(.text_caller_toc) }
}

#--- ldsswap
SECTIONS {
  .text_func         0x1001000 : { *(.text_func) }
  .text_callee       0x1002000 : { *(.text_callee) }
  .text_caller_toc   0x10030000 : { *(.text_caller_toc) }
  .text_caller_notoc 0x10040000 : { *(.text_caller_notoc) }
}

#--- callees.s
.section .text_func, "ax", %progbits
func:
  blr

.globl callee_toc
.section .text_callee, "ax", %progbits
callee_toc:
.Lfunc_gep1:
  addis 2, 12, .TOC.-.Lfunc_gep1@ha
  addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
  .localentry callee_toc, .Lfunc_lep1-.Lfunc_gep1
  bl func
  addis 4, 2, global@toc@ha
  lwz 4, global@toc@l(4)
  blr

.globl callee_notoc
callee_notoc:
  .localentry callee_notoc, 1
  bl func
  plwz 4, global@pcrel(0), 1
  blr

## .globl global
global:
  .long	0
  .size	global, 4

#--- callernotoc.s
.section .text_caller_notoc, "ax", %progbits
caller_notoc:
  .localentry caller, 1
  bl callee_toc@notoc
  bl callee_notoc@notoc
  blr

#--- callertoc.s
.section .text_caller_toc, "ax", %progbits
caller_toc:
.Lfunc_gep2:
  addis 2, 12, .TOC.-.Lfunc_gep2@ha
  addi 2, 2, .TOC.-.Lfunc_gep2@l
.Lfunc_lep2:
  .localentry caller_toc, .Lfunc_lep2-.Lfunc_gep2
  bl callee_toc
  nop
  bl callee_notoc
  nop
  blr
