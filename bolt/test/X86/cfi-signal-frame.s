## Check that BOLT preserves the signal frame CIE augmentation ("S") of
## functions marked with .cfi_signal_frame.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0
# RUN: llvm-nm %t.bolt > %t.txt
# RUN: llvm-dwarfdump --eh-frame %t.bolt >> %t.txt
# RUN: FileCheck %s --input-file %t.txt

# CHECK: [[#%x,TRAMP:]] T tramp

## The first CIE belongs to the rewritten .eh_frame.
# CHECK:      [[CIE:[0-9a-f]+]] {{[0-9a-f]+}} 00000000 CIE
# CHECK-NEXT:   Format:
# CHECK-NEXT:   Version:
# CHECK-NEXT:   Augmentation: "zRS"
# CHECK:      FDE cie=[[CIE]] pc=[[#%.8x,TRAMP]]...

  .text
  .globl _start
  .type _start, %function
_start:
  leaq tramp(%rip), %rax
  pushq %rax
  ret
  .size _start, .-_start

  .globl tramp
  .type tramp, %function
tramp:
  .cfi_startproc
  .cfi_signal_frame
  xorl %edi, %edi
  movl $60, %eax
  syscall
  .cfi_endproc
  .size tramp, .-tramp
