# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/so.s -o %t/so.o
# RUN: ld.lld -shared %t/so.o -o %t/libso.so

# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t/main.o
# RUN: ld.lld --image-base=0x1000 --section-start=.text=0x1000 %t/main.o %t/libso.so -o %t/out
# RUN: llvm-readelf -x .text %t/out | FileCheck %s

# CHECK: Hex dump of section '.text':
# CHECK-NEXT: 0x00001000 0c000000 00000000 00000000 c3

#--- so.s
.global shared_sym
shared_sym:
  ret

#--- main.s
.globl _start
_start:
  # R_X86_64_PCNEXT32 to defined local symbol at 0x100c (place is 0x1000, 0x100c - 0x1000 = 0x0c)
  .reloc ., R_X86_64_PCNEXT32, local_sym
  .byte 0x00, 0x00, 0x00, 0x00

  # R_X86_64_PCNEXT32 to shared symbol (resolves to 0)
  .reloc ., R_X86_64_PCNEXT32, shared_sym
  .byte 0x00, 0x00, 0x00, 0x00

  # R_X86_64_PCNEXT32 to undefined weak symbol (resolves to 0)
  .reloc ., R_X86_64_PCNEXT32, undef_sym
  .byte 0x00, 0x00, 0x00, 0x00

local_sym:
  ret

.weak undef_sym
