## Check that llvm-bolt will not unnecessarily relax ADR instruction.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=random2
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt | FileCheck %s
# RUN: llvm-objdump -d --disassemble-symbols=foo.cold.0 %t.bolt \
# RUN:   | FileCheck --check-prefix=CHECK-FOO %s

## ADR below references its containing function that is split. But ADR is always
## in the main fragment, thus there is no need to relax it.
  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: <_start>:
  .cfi_startproc
  adr x1, _start
# CHECK-NOT: adrp
# CHECK: adr
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
  mov  x0, #0x0
.L2:
# CHECK-FOO: <foo.cold.0>:
  adr x1, foo
# CHECK-FOO: adrp
# CHECK-FOO-NEXT: add
  ret  x30
  .cfi_endproc
.size foo, .-foo

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
