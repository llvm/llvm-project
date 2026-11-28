# REQUIRES: sparc
## SPARC uses Elf64_Rela relocation entries. With -z rel the addends of the
## dynamic relocations are written to the relocated locations instead, and
## --check-dynamic-relocations reads them back.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparcv9 -position-independent a.s -o a.o
# RUN: ld.lld -shared -z rel --check-dynamic-relocations a.o -o a.so
# RUN: llvm-readobj -r -x .data -x .got a.so | FileCheck %s

# CHECK:      .rel.dyn {
# CHECK-NEXT:   0x[[#%x,DATA:]] R_SPARC_RELATIVE -
# CHECK-NEXT:   R_SPARC_64 _start
# CHECK-NEXT:   R_SPARC_GLOB_DAT func
# CHECK-NEXT:   R_SPARC_TLS_TPOFF64 tv
# CHECK-NEXT: }
# CHECK-NEXT: .rel.plt {
# CHECK-NEXT:   R_SPARC_JMP_SLOT func
# CHECK-NEXT: }

## The GOT header is followed by the R_SPARC_GLOB_DAT and R_SPARC_TLS_TPOFF64
## slots, whose addends are 0. The R_SPARC_RELATIVE addend is the address of
## .data and the R_SPARC_64 addend is 42.

# CHECK:      Hex dump of section '.got':
# CHECK-NEXT: 0x{{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} 00000000 00000000
# CHECK-NEXT: 0x{{[0-9a-f]+}} 00000000 00000000

# CHECK:      Hex dump of section '.data':
# CHECK-NEXT: 0x[[#%.8x,DATA]] 00000000 [[#%.8x,DATA]] 00000000 0000002a

## R_SPARC_COPY has no implicit addend.
# RUN: llvm-mc -filetype=obj -triple=sparcv9 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=sparcv9 -position-independent c.s -o c.o
# RUN: ld.lld -shared c.o -soname=c.so -o c.so
# RUN: ld.lld -z rel --check-dynamic-relocations b.o c.so -o b
# RUN: llvm-readobj -r b | FileCheck --check-prefix=COPY %s

# COPY:      .rel.dyn {
# COPY-NEXT:   R_SPARC_COPY obj
# COPY-NEXT: }

#--- a.s
.globl _start, func
.type func,@function
_start:
func:
  call  func
  nop
  sethi %got22(func), %g1
  or    %g1, %got10(func), %g1
  sethi %tie_hi22(tv), %g1
  add   %g1, %tie_lo10(tv), %g1

.data
  .xword .data
  .xword _start+42

.section .tbss,"awT",@nobits
.globl tv
tv:
  .skip 8

#--- b.s
.globl _start
_start:
  sethi %hi(obj), %g1
  or    %g1, %lo(obj), %g1

#--- c.s
.data
.globl obj
.type obj,@object
.size obj,8
obj:
  .xword 1
