## Check that llvm-bolt successfully relaxes branches for compact (<128MB) code
## model.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=randomN \
# RUN:   --keep-nops --compact-code-model
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt | FileCheck %s
# RUN: llvm-nm -n %t.bolt | FileCheck %s --check-prefix=CHECK-NM

## _start will be split and its main fragment will be separated from other
## fragments by large_function() which is over 1MB.

# CHECK-NM: _start
# CHECK-NM-NEXT: large_function
# CHECK-NM-NEXT: _start.cold

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  cmp  x1, 1
  b.hi  .L1
# CHECK: b.hi
# CHECK-NEXT: b
# CHECK-NEXT: b

  bl large_function
.L1:
  ret  x30
  .cfi_endproc
.size _start, .-_start


  .globl large_function
  .type large_function, %function
large_function:
  .cfi_startproc
  .rept 300000
    nop
  .endr
  ret  x30
  .cfi_endproc
.size large_function, .-large_function

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
