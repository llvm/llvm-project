# REQUIRES: x86

## On RELA targets, r_addend may be updated for --emit-relocs.
## With ICF, merged sections do not have output sections assigned.
## Test we don't crash.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --gc-sections --emit-relocs --icf=all %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section (3) .rela.text {
# CHECK-NEXT:     R_X86_64_64 .text 0x11
# CHECK-NEXT:     R_X86_64_64 .text 0x11
# CHECK-NEXT:     R_X86_64_64 .rodata 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.rodata
quux:
.quad 0xfe

.section .text.foo,"ax"
foo:
.quad quux

.section .text.bar,"ax"
bar:
.quad quux

.section .text.baz,"ax"
baz:
.quad quux

.text
.quad foo
.quad bar

.global _start
_start:
  nop
