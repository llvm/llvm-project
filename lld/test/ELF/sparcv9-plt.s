# REQUIRES: sparc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparcv9 b.s -o b.o
# RUN: ld.lld -shared b.o -soname=b.so -o b.so
# RUN: llvm-mc --position-independent -filetype=obj -triple=sparcv9 a.s -o a.o
# RUN: ld.lld a.o b.so -o a
# RUN: llvm-readelf -S -d -r -s a | FileCheck %s --implicit-check-not=.got.plt
# RUN: llvm-objdump -d -z -j .text -j .plt --no-print-imm-hex a | FileCheck --check-prefix=DIS %s

## SPARC has no .got.plt. The dynamic linker rewrites the PLT entries in place,
## so .plt is writable and DT_PLTGOT points at it, and .rela.plt names .plt in
## its sh_info.
# CHECK: [ 5] .rela.plt RELA {{[0-9a-f]+}} {{[0-9a-f]+}} 000048 18 AI 1 7 8
# CHECK: [ 7] .plt PROGBITS 0000000000300320 {{[0-9a-f]+}} 0000e0 00 WAX
# CHECK: (PLTGOT) 0x300320

## R_SPARC_JMP_SLOT applies to the PLT entry rather than to a .got.plt slot.
# CHECK:      Relocation section '.rela.plt' {{.*}} contains 3 entries:
# CHECK:      00000000003003a0 {{.*}} R_SPARC_JMP_SLOT {{.*}} weak + 0
# CHECK-NEXT: 00000000003003c0 {{.*}} R_SPARC_JMP_SLOT {{.*}} bar + 0
# CHECK-NEXT: 00000000003003e0 {{.*}} R_SPARC_JMP_SLOT {{.*}} foo + 0

# CHECK:      0000000000000000 0 FUNC WEAK   DEFAULT UND weak
# CHECK-NEXT: 0000000000000000 0 FUNC GLOBAL DEFAULT UND bar
# CHECK-NEXT: 00000000003003e0 0 FUNC GLOBAL DEFAULT UND foo

# DIS:      <_start>:
# DIS-NEXT:   call 0x3003c0
# DIS-NEXT:   nop
# DIS-NEXT:   call 0x3003a0
# DIS-NEXT:   nop

## The four reserved entries are left zeroed, as GNU ld leaves them. The
## dynamic linker writes the resolver code into .PLT0 and .PLT1 at startup.
# DIS:          0000000000300320 <.plt>:
# DIS-COUNT-32:   unimp 0

## Each entry puts its own offset from .plt into %g1 and branches to .PLT1,
## which the dynamic linker redirects to the resolver.
# DIS-NEXT:   3003a0: 03 00 00 80 sethi 128, %g1
# DIS-NEXT:           30 6f ff e7 ba,a %xcc, 0x300340
# DIS:        3003c0: 03 00 00 a0 sethi 160, %g1
# DIS-NEXT:           30 6f ff df ba,a %xcc, 0x300340
# DIS:        3003e0: 03 00 00 c0 sethi 192, %g1
# DIS-NEXT:           30 6f ff d7 ba,a %xcc, 0x300340

#--- a.s
.globl _start
.weak weak
_start:
  call bar
  nop
  call weak
  nop
## Taking foo's address forces a canonical PLT.
  sethi %pc22(foo), %g1
  or    %g1, %pc10(foo), %g1

#--- b.s
.globl bar, weak, foo
.type bar,@function
.type weak,@function
.type foo,@function
bar:
weak:
foo:
