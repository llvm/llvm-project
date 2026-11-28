## Check that ICF does not fold functions that have identical code and CFI but
## different exception handling state: the personality routine, whether the
## function has an LSDA at all, and the DW_CFA_GNU_args_size value at a call
## with a landing pad. Functions whose personality routines are different but
## identical will be folded once the personality routines have folded.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --icf=all 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-ICF
# RUN: llvm-nm -n %t.bolt > %t.txt
# RUN: llvm-dwarfdump --eh-frame %t.bolt >> %t.txt
# RUN: FileCheck %s --input-file %t.txt
# RUN: FileCheck %s --input-file %t.txt --check-prefix=CHECK-FOLD

## same_b is folded into same_a, pers_d into pers_c and then fold_y into
## fold_x.
# CHECK-ICF: BOLT-INFO: ICF folded 3 out of {{[0-9]+}} functions

# CHECK-DAG: [[#%x,PERS_A:]] T pers_a
# CHECK-DAG: [[#%x,PERS_B:]] T pers_b
# CHECK-DAG: [[#%x,PERS_X:]] T pers_x
# CHECK-DAG: [[#%x,PERS_Y:]] T pers_y
# CHECK-DAG: [[#%x,SAME:]] T same_a
# CHECK-DAG: [[#SAME]] T same_b

## pers_x and pers_y keep their own personality routines: each FDE follows a
## CIE that names the right one.
# CHECK:      Personality Address: [[#%.16x,PERS_A]]
# CHECK-NOT:  CIE
# CHECK:      FDE cie={{[0-9a-f]+}} pc=[[#%.8x,PERS_X]]...
# CHECK:      Personality Address: [[#%.16x,PERS_B]]
# CHECK-NOT:  CIE
# CHECK:      FDE cie={{[0-9a-f]+}} pc=[[#%.8x,PERS_Y]]...

## fold_x and fold_y share one FDE, which names the folded personality routine.
# CHECK-FOLD-DAG: [[#%x,PERS_C:]] T pers_c
# CHECK-FOLD-DAG: [[#PERS_C]] T pers_d
# CHECK-FOLD-DAG: [[#%x,FOLD:]] T fold_x
# CHECK-FOLD-DAG: [[#FOLD]] T fold_y
# CHECK-FOLD:     Personality Address: [[#%.16x,PERS_C]]
# CHECK-FOLD-NOT: CIE
# CHECK-FOLD:     FDE cie={{[0-9a-f]+}} pc=[[#%.8x,FOLD]]...

  .text
  .globl _start
  .type _start, %function
_start:
  call same_a
  call same_b
  call pers_x
  call pers_y
  call fold_x
  call fold_y
  call lsda_x
  call lsda_y
  call args_x
  call args_y
  xorl %edi, %edi
  movl $60, %eax
  syscall
  .size _start, .-_start

  .globl callee
  .type callee, %function
callee:
  ret
  .size callee, .-callee

  .globl resume
  .type resume, %function
resume:
  ud2
  .size resume, .-resume

  .globl pers_a
  .type pers_a, %function
pers_a:
  movl $1, %eax
  ret
  .size pers_a, .-pers_a

  .globl pers_b
  .type pers_b, %function
pers_b:
  movl $2, %eax
  ret
  .size pers_b, .-pers_b

## Two identical personality routines.
  .globl pers_c
  .type pers_c, %function
pers_c:
  movl $3, %eax
  ret
  .size pers_c, .-pers_c

  .globl pers_d
  .type pers_d, %function
pers_d:
  movl $3, %eax
  ret
  .size pers_d, .-pers_d

## A call with a cleanup landing pad. Argument n tells the copies apart.
  .macro INVOKE name, pers, n
  .globl \name
  .type \name, %function
\name:
  .cfi_startproc
  .cfi_personality 0x1b, \pers
  .cfi_lsda 0x1b, .Lexc_\name
  pushq %rax
  .cfi_def_cfa_offset 16
  movl $\n, %edi
.Lbegin_\name:
  call callee
.Lend_\name:
  popq %rax
  .cfi_def_cfa_offset 8
  retq
.Llp_\name:
  .cfi_def_cfa_offset 16
  movq %rax, %rdi
  call resume
  .cfi_endproc
  .size \name, .-\name

  .pushsection .gcc_except_table,"a",@progbits
.Lexc_\name:
  .byte 0xff
  .byte 0xff
  .byte 0x01
  .uleb128 .Lcst_end_\name - .Lcst_\name
.Lcst_\name:
  .uleb128 .Lbegin_\name - \name
  .uleb128 .Lend_\name - .Lbegin_\name
  .uleb128 .Llp_\name - \name
  .byte 0
.Lcst_end_\name:
  .popsection
  .endm

## Identical code and EH state: folded.
  INVOKE same_a, pers_a, 1
  INVOKE same_b, pers_a, 1

## Different personality routines.
  INVOKE pers_x, pers_a, 2
  INVOKE pers_y, pers_b, 2

## Different personality routines that fold.
  INVOKE fold_x, pers_c, 4
  INVOKE fold_y, pers_d, 4

## An LSDA with an empty call site table: if callee throws, the program
## terminates. Without an LSDA, the exception propagates.
  .globl lsda_x
  .type lsda_x, %function
lsda_x:
  .cfi_startproc
  .cfi_personality 0x1b, pers_a
  .cfi_lsda 0x1b, .Lexc_lsda_x
  pushq %rax
  .cfi_def_cfa_offset 16
  movl $3, %edi
  call callee
  popq %rax
  .cfi_def_cfa_offset 8
  retq
  .cfi_endproc
  .size lsda_x, .-lsda_x

  .globl lsda_y
  .type lsda_y, %function
lsda_y:
  .cfi_startproc
  .cfi_personality 0x1b, pers_a
  pushq %rax
  .cfi_def_cfa_offset 16
  movl $3, %edi
  call callee
  popq %rax
  .cfi_def_cfa_offset 8
  retq
  .cfi_endproc
  .size lsda_y, .-lsda_y

  .pushsection .gcc_except_table,"a",@progbits
.Lexc_lsda_x:
  .byte 0xff
  .byte 0xff
  .byte 0x01
  .uleb128 0
  .popsection

## Different DW_CFA_GNU_args_size at the call.
  .macro ARGS name, size
  .globl \name
  .type \name, %function
\name:
  .cfi_startproc
  .cfi_personality 0x1b, pers_a
  .cfi_lsda 0x1b, .Lexc_\name
  pushq %rax
  .cfi_def_cfa_offset 16
  pushq $1
  .cfi_def_cfa_offset 24
  pushq $2
  .cfi_def_cfa_offset 32
  .cfi_escape 0x2e, \size
.Lbegin_\name:
  call callee
.Lend_\name:
  addq $24, %rsp
  .cfi_def_cfa_offset 8
  retq
.Llp_\name:
  .cfi_def_cfa_offset 32
  movq %rax, %rdi
  call resume
  .cfi_endproc
  .size \name, .-\name

  .pushsection .gcc_except_table,"a",@progbits
.Lexc_\name:
  .byte 0xff
  .byte 0xff
  .byte 0x01
  .uleb128 .Lcst_end_\name - .Lcst_\name
.Lcst_\name:
  .uleb128 .Lbegin_\name - \name
  .uleb128 .Lend_\name - .Lbegin_\name
  .uleb128 .Llp_\name - \name
  .byte 0
.Lcst_end_\name:
  .popsection
  .endm

  ARGS args_x, 0x10
  ARGS args_y, 0x00
