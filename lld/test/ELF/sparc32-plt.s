# REQUIRES: sparc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparc a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=sparc b.s -o b.o
# RUN: ld.lld -shared b.o -soname=b.so -o b.so
# RUN: ld.lld a.o b.so -o a
# RUN: llvm-readelf -S -r a | FileCheck %s
# RUN: llvm-objdump -d -j .plt --no-show-raw-insn a | FileCheck %s --check-prefix=PLT

## As on V9 the loader resolves a PLT slot by writing a jump into the entry, so
## .plt is writable and the JMP_SLOT relocation names the entry rather than a
## .got.plt slot. The 32-bit entry is 12 bytes and the reserved header is four
## of them, so the first entry is at .plt + 0x30.
# CHECK:      .plt PROGBITS [[#%x,PLTADDR:]] {{.*}} WAX
# CHECK:      Relocation section '.rela.plt' {{.*}} contains 1 entries:
# CHECK:      {{0*}}[[#%x,PLTADDR+0x30]] {{.*}} R_SPARC_JMP_SLOT {{.*}} g + 0

## The sethi carries the entry's offset from .PLT0, which is how the resolver
## recovers the relocation index, and the branch goes back to .PLT0.
# PLT:      <.plt>:
# PLT:      sethi 0x30, %g1
# PLT-NEXT: ba,a
# PLT-NEXT: nop

#--- a.s
.globl _start
_start:
  call g
  nop

#--- b.s
.globl g
.type g,@function
g:
  retl
  nop
