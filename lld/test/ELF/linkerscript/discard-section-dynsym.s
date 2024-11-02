# REQUIRES: aarch64

## We allow discarding .dynsym, check we don't crash.
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o

# RUN: echo 'SECTIONS { /DISCARD/ : { *(.dynsym) } }' > %t.lds
# RUN: ld.lld -shared -T %t.lds %t.o -o %t.so
# RUN: llvm-readelf -r %t.so | FileCheck %s

# RUN: echo 'SECTIONS { /DISCARD/ : { *(.dynsym .dynstr) } }' > %t.lds
# RUN: ld.lld -shared -T %t.lds %t.o -o %t.so
# RUN: llvm-readelf -r %t.so | FileCheck %s

# CHECK:      contains 2 entries:
# CHECK:      R_AARCH64_RELATIVE  [[#]]
# CHECK-NEXT: R_AARCH64_GLOB_DAT  0{{$}}

  adrp x9, :got:var
  ldr  x9, [x9, :got_lo12:var]

.data
.align 8
foo:
.quad foo
