# REQUIRES: riscv
# RUN: llvm-mc -triple=riscv32 -mattr=+c,+relax -filetype=obj %s -o %t.32.o
# RUN: ld.lld --relax -Ttext=0x20000 -e _start %t.32.o -o %t.32
# RUN: llvm-nm --defined-only %t.32 | FileCheck %s
# RUN: ld.lld --no-relax -Ttext=0x20000 -e _start %t.32.o -o %t.32.no-relax
# RUN: llvm-nm --defined-only %t.32.no-relax | FileCheck %s

# RUN: llvm-mc -triple=riscv64 -mattr=+c,+relax -filetype=obj %s -o %t.64.o
# RUN: ld.lld --relax -Ttext=0x20000 -e _start %t.64.o -o %t.64
# RUN: llvm-nm --defined-only %t.64 | FileCheck %s
# RUN: ld.lld --no-relax -Ttext=0x20000 -e _start %t.64.o -o %t.64.no-relax
# RUN: llvm-nm --defined-only %t.64.no-relax | FileCheck %s

## Also check an object assembled without relaxation.
# RUN: llvm-mc -triple=riscv32 -mattr=+c,-relax -filetype=obj %s -o %t.fixed.o
# RUN: ld.lld --relax -Ttext=0x20000 -e _start %t.fixed.o -o %t.fixed
# RUN: llvm-nm --defined-only %t.fixed | FileCheck %s

## Preferred-alignment padding must remain adjustable when preceding code
## can shrink, so the actual function entry retains its required alignment.
## This body selects 64-byte alignment, which also satisfies .p2align 2.
##
## Alignment relocations are processed even with linker --no-relax.
# CHECK: {{[048c]0}} T big{{$}}

.text
.globl _start, big

.type _start, @function
_start:
  call big
.size _start, .-_start

.p2align 2
.prefalign 6, .Lbig_end, nop
.type big, @function
big:
.rept 31
  c.nop
.endr
  ret
.Lbig_end:
.size big, .Lbig_end-big
