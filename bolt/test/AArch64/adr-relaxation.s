## Check that llvm-bolt will not unnecessarily relax ADR instruction.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=random2
# RUN: llvm-objdump -d -j .text %t.bolt | FileCheck %s
# RUN: llvm-objdump -d --disassemble-symbols=foo.cold.0 %t.bolt \
# RUN:   | FileCheck --check-prefix=CHECK-FOO %s

## ADR below references its containing function that is split. But ADR is always
## in the main fragment, thus there is no need to relax it.
  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# CHECK: <_start>:
# CHECK-NEXT: adr
  adr x1, _start
  cmp  x1, x11
  b.hi  .L1
  mov  x0, #0x0
.L1:
  ret  x30
  .cfi_endproc
.size _start, .-_start


## In foo, ADR is in the split fragment but references the main one. Thus, it
## needs to be relaxed into ADRP + ADD.
  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  cmp  x1, x11
  b.hi  .L2
.L2:
# CHECK-FOO: <foo.cold.0>:
# CHECK-FOO-NEXT: adrp
# CHECK-FOO-NEXT: add
  adr x1, foo
  ret  x30
  .cfi_endproc
.size foo, .-foo

## bar is a non-simple function. We can still relax ADR, because it has a
## preceding NOP.
  .globl bar
  .type bar, %function
bar:
  .cfi_startproc
# CHECK-LABEL: <bar>:
# CHECK-NEXT: adrp
# CHECK-NEXT: add
  nop
  adr x0, foo
  adr x1, .L3
  br x1
.L3:
  ret  x0
  .cfi_endproc
  .size bar, .-bar
