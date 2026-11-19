## Conservative mode updates pointer arrays only. Aggressive mode also treats
## matching aligned words in selected general data sections as pointers.
# RUN: echo target > %t.order
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.conservative --recover-relocations \
# RUN:   --reorder-functions=user --function-order=%t.order
# RUN: %python %p/../Inputs/check-recovered-addresses.py x86-conservative \
# RUN:   %t.exe %t.conservative
# RUN: not llvm-bolt %t.exe -o %t.bad \
# RUN:   --aggressive-relocation-recovery \
# RUN:   2>&1 | FileCheck %s --check-prefix=OPTION
# RUN: llvm-bolt %t.exe -o %t.bolt --recover-relocations \
# RUN:   --aggressive-relocation-recovery \
# RUN:   --reorder-functions=user --function-order=%t.order
# RUN: %python %p/../Inputs/check-recovered-addresses.py x86 %t.exe %t.bolt
# RUN: llvm-nm %t.bolt | FileCheck %s --check-prefix=SYMBOL
# OPTION: --aggressive-relocation-recovery requires --recover-relocations
# SYMBOL: T target

.text
.globl _start
.type _start, @function
_start:
.cfi_startproc
  call target
  ret
.cfi_endproc
.size _start, .-_start
.globl target
.type target, @function
target:
.cfi_startproc
  ret
.cfi_endproc
.size target, .-target
.data
.p2align 3
pointer:
  .quad target
## Short tail must not be read as a complete pointer.
  .byte 1, 2, 3
.section .init_array,"aw",@init_array
.p2align 3
  .quad target
