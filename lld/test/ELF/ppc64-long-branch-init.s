# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64-pc-freebsd13.0 %s -o %t.o
# RUN: echo 'SECTIONS { .text 0x10000 : { *(.text) } .init 0x2010000 : { *(.init) } }' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## .init consists of sections from several object files. Sections other than the
## last one do not have a terminator. Check we do not create a long branch stub
## in the middle.
## We currently use thunk section spacing to ensure the stub is in the end. This
## is not foolproof but good enough to not break in practice.

# CHECK: Disassembly of section .init:
# CHECK-EMPTY:
# CHECK-LABEL: <_init>:
# CHECK:         blr
# CHECK-NEXT:    trap
# CHECK-EMPTY:
# CHECK-LABEL: <__long_branch_foo>:

.globl foo
foo:
  blr

.section .init,"ax",@progbits,unique,0
.globl _init
_init:
  stdu 1, -48(1)
  mflr 0
  std 0, 64(1)

.section .init,"ax",@progbits,unique,1
  bl foo
  nop

.section .init,"ax",@progbits,unique,2
  bl foo
  nop

.section .init,"ax",@progbits,unique,3
  ld 1, 0(1)
  ld 0, 16(1)
  mtlr 0
  blr
