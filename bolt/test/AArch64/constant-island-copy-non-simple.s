## Check that with --clone-constant-island=0, a non-simple function whose LDR
## literal or ADR referencing another function's constant island cannot be
## relaxed gets a copy of that island instead of failing. A simple function
## loading from the same island is still relaxed into ADRP+LDR.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.bolt --clone-constant-island=0 2>&1 \
# RUN:   | FileCheck %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-DIS

# CHECK: internal call detected in function nonsimple
# CHECK-NOT: BOLT-ERROR
# CHECK: BOLT-INFO: copied 1 constant island(s) totaling 4 bytes into non-simple functions
# CHECK-NOT: BOLT-ERROR

## The LDR and ADR in nonsimple reference the copy emitted after nonsimple's
## code. Island copies get no $d mapping symbol, so the copy may be shown as an
## instruction.
# CHECK-DIS-LABEL: <nonsimple>:
# CHECK-DIS:         ldr w0, 0x[[#%x,COPY:]] <nonsimple+0x{{[0-9a-f]+}}>
# CHECK-DIS-NEXT:    adr x1, 0x[[#%x,COPY]] <nonsimple+0x{{[0-9a-f]+}}>
# CHECK-DIS:         [[#%x,COPY]]: {{12345678|78 56 34 12}}
# CHECK-DIS-LABEL: <simple>:
# CHECK-DIS-NEXT:    adrp x0
# CHECK-DIS-NEXT:    ldr w0, [x0

  .text
  .globl _start
  .type _start, %function
_start:
  bl nonsimple
  bl simple
  bl owner
  ret
  .size _start, .-_start

## The internal call makes nonsimple non-simple, so BOLT keeps its layout and
## cannot grow the LDR into ADRP+LDR or the ADR into ADRP+ADD.
  .globl nonsimple
  .type nonsimple, %function
nonsimple:
  stp x29, x30, [sp, #-16]!
  bl .Lsub
  ldp x29, x30, [sp], #16
  ret
.Lsub:
  ldr w0, .Lpool
  adr x1, .Lpool
  ret
  .size nonsimple, .-nonsimple

  .globl simple
  .type simple, %function
simple:
  ldr w0, .Lpool
  ret
  .size simple, .-simple

## The literal pool follows owner's code but lies outside its symbol size, where
## an assembler emits pending literals at the end of a section.
  .globl owner
  .type owner, %function
owner:
  ldr w0, .Lpool
  ret
  .size owner, .-owner
.Lpool:
  .word 0x12345678

  .globl next
  .type next, %function
next:
  ret
  .size next, .-next
