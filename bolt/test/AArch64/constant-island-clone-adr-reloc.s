## Check that an ADR referencing another function's constant island through a
## relocation gets a copy of the island with the default --clone-constant-island,
## like a reference symbolized without a relocation, and keeps referencing the
## original island with --clone-constant-island=0.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CLONE
# RUN: llvm-bolt %t.exe -o %t.noclone.bolt --clone-constant-island=0
# RUN: llvm-objdump -d %t.noclone.bolt | FileCheck %s --check-prefix=NOCLONE

## With cloning, the ADR reads a copy emitted after caller's code.
# CLONE-LABEL: <caller>:
# CLONE-NEXT:    adr x0, 0x[[#%x,COPY:]] <caller+0x{{[0-9a-f]+}}>
# CLONE:         [[#%x,COPY]]: {{12345678|78 56 34 12}}

## Without cloning, the ADR is relaxed to reach the island in owner.
# NOCLONE-LABEL: <caller>:
# NOCLONE-NEXT:    adrp x0
# NOCLONE-NEXT:    add x0, x0

  .text
  .globl _start
  .type _start, %function
_start:
  bl caller
  bl owner
  ret
  .size _start, .-_start

## owner is a global symbol, so the assembler keeps an R_AARCH64_ADR_PREL_LO21
## relocation for the ADR against owner+8, the literal after owner's code.
  .globl caller
  .type caller, %function
caller:
  adr x0, owner + 8
  ret
  .size caller, .-caller

  .globl owner
  .type owner, %function
owner:
  adr x0, .Lpool
  ret
  .size owner, .-owner
.Lpool:
  .word 0x12345678

  .globl next
  .type next, %function
next:
  ret
  .size next, .-next
