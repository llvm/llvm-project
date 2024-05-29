# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld %t.o -o %t.exe -q --unresolved-symbols=ignore-all
# RUN: llvm-readelf -Wr %t.exe | FileCheck %s
# RUN: llvm-bolt --strict %t.exe --relocs -o %t.null

  .text
  .globl _start
  .type _start,@function
_start:
  .cfi_startproc
  retq

# For relocations against .text
  call exit
  .size _start, .-_start
  .cfi_endproc

  .data
var:
  .quad 0

  .rodata
var_offset64:
  .quad var-.
var_offset32:
  .long var-.
var_offset16:
  .word var-.

## Check that BOLT succeeds in strict mode in the presence of unaccounted
## data-to-data PC-relative relocations.

# CHECK: Relocation section '.rela.rodata'
# CHECK-NEXT: Offset
# CHECK-NEXT: R_X86_64_PC64
# CHECK-NEXT: R_X86_64_PC32
# CHECK-NEXT: R_X86_64_PC16
