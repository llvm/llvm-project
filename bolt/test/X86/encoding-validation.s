# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe --relocs -o %t.out --check-encoding |& FileCheck %s

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc

## Check that llvm-bolt uses non-symbolizing disassembler while validating
## instruction encodings. If symbol "foo" below is symbolized, the encoded
## instruction would have a different sequence of bytes from the input
## sequence, as "foo" will not have any address assigned at that point.

  movq foo(%rip), %rax
# CHECK-NOT: mismatching LLVM encoding detected

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
