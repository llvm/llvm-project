# REQUIRES: system-linux

## Check that BOLT recognizes PT_GNU_RELRO segment and marks respective sections
## accordingly.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --no-relax
# RUN: llvm-readelf -We %t.exe | FileCheck --check-prefix=READELF %s
# Unfortunately there's no direct way to extract a segment to section mapping
# for a given section from readelf. Use the fool-proof way of matching
# readelf output line-by-line.
# READELF:      Program Headers:
# READELF-NEXT: Type Offset {{.*}}
# READELF-NEXT: PHDR
# READELF-NEXT: LOAD
# READELF-NEXT: LOAD
# READELF-NEXT: LOAD
# READELF-NEXT: GNU_RELRO
# (GNU_RELRO is segment 4)

# READELF: Section to Segment mapping:
# READELF: 04 .got

# RUN: llvm-bolt %t.exe --relocs -o %t.null -v=1 \
# RUN:   2>&1 | FileCheck --check-prefix=BOLT %s
# BOLT: BOLT-INFO: marking .got as GNU_RELRO

  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  jmp *foo@GOTPCREL(%rip)
  ret
  .cfi_endproc
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo
