# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 -crel a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 -crel b.s -o b.o
# RUN: ld.lld -r b.o a.o -o out
# RUN: llvm-readobj -r out | FileCheck %s --check-prefixes=CHECK,CRELFOO

# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a1.o
# RUN: ld.lld -r b.o a1.o -o out1
# RUN: llvm-readobj -r out1 | FileCheck %s --check-prefixes=CHECK,RELAFOO
# RUN: ld.lld -r a1.o b.o -o out2
# RUN: llvm-readobj -r out2 | FileCheck %s --check-prefixes=CHECK2

# CHECK:      Relocations [
# CHECK-NEXT:   .crel.text {
# CHECK-NEXT:     0x1 R_X86_64_PLT32 fb 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:     0x9 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:     0xE R_X86_64_PLT32 .text.foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   }
# CHECK-NEXT:   .crel.rodata {
# CHECK-NEXT:     0x0 R_X86_64_PC32 foo 0x0
# CHECK-NEXT:     0xF R_X86_64_PC32 foo 0x3F
# CHECK-NEXT:     0x1F R_X86_64_PC64 foo 0x7F
# CHECK-NEXT:     0x27 R_X86_64_PC32 _start 0xFFFFFFFFFFFFE07F
# CHECK-COUNT-12:      R_X86_64_32 _start 0x0
# CHECK-NEXT:   }
# CRELFOO-NEXT: .crel.text.foo {
# RELAFOO-NEXT: .rela.text.foo {
# CHECK-NEXT:     0x3 R_X86_64_PC32 .L.str 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:     0xA R_X86_64_PC32 .L.str1 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:     0xF R_X86_64_PLT32 g 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:     0x14 R_X86_64_PLT32 g 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   }
# CRELFOO-NEXT: .crel.data {
# RELAFOO-NEXT: .rela.data {
# CHECK-NEXT:     0x8 R_X86_64_64 _start 0x8000000000000000
# CHECK-NEXT:     0x18 R_X86_64_64 _start 0xFFFFFFFFFFFFFFFF
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECK2:      Relocations [
# CHECK2-NEXT:   .crel.text {
# CHECK2-NEXT:     0x1 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
# CHECK2-NEXT:     0x6 R_X86_64_PLT32 .text.foo 0xFFFFFFFFFFFFFFFC
# CHECK2-NEXT:     0xD R_X86_64_PLT32 fb 0xFFFFFFFFFFFFFFFC
# CHECK2-NEXT:   }
# CHECK2-NEXT:   .rela.text.foo {
# CHECK2-NEXT:     0x3 R_X86_64_PC32 .L.str 0xFFFFFFFFFFFFFFFC
# CHECK2-NEXT:     0xA R_X86_64_PC32 .L.str1 0xFFFFFFFFFFFFFFFC
# CHECK2-NEXT:     0xF R_X86_64_PLT32 g 0xFFFFFFFFFFFFFFFC
# CHECK2-NEXT:     0x14 R_X86_64_PLT32 g 0xFFFFFFFFFFFFFFFC
# CHECK2-NEXT:   }
# CHECK2-NEXT:   .rela.data {
# CHECK2-NEXT:     0x8 R_X86_64_64 _start 0x8000000000000000
# CHECK2-NEXT:     0x18 R_X86_64_64 _start 0xFFFFFFFFFFFFFFFF
# CHECK2-NEXT:   }
# CHECK2-NEXT:   .crel.rodata {
# CHECK2-NEXT:     0x0 R_X86_64_PC32 foo 0x0
# CHECK2-NEXT:     0xF R_X86_64_PC32 foo 0x3F
# CHECK2-NEXT:     0x1F R_X86_64_PC64 foo 0x7F
# CHECK2-NEXT:     0x27 R_X86_64_PC32 _start 0xFFFFFFFFFFFFE07F
# CHECK2-COUNT-12:      R_X86_64_32 _start 0x0
# CHECK2-NEXT:   }
# CHECK2-NEXT: ]

#--- a.s
.global _start, foo
_start:
  call foo
  call .text.foo

.section .text.foo,"ax"
foo:
  leaq .L.str(%rip), %rsi
  leaq .L.str1(%rip), %rsi
  call g
  call g

.section .rodata.str1.1,"aMS",@progbits,1
.L.str:
  .asciz  "abc"
.L.str1:
  .asciz  "def"

.data
.quad 0
.quad _start - 0x8000000000000000
.quad 0
.quad _start - 1

#--- b.s
.globl fb
fb:
  call fb

.section .rodata,"a"
.long foo - .
.space 15-4
.long foo - . + 63  # offset+=15
.space 16-4
.quad foo - . + 127  # offset+=16
.long _start - . - 8065

## Ensure .crel.rodata contains 16 relocations so that getULEB128Size(crelHeader) > 1.
.rept 12
.long _start
.endr
