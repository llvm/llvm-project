## Check that BOLT can correctly use relocations to symbolize instruction
## operands when an instruction can have up to two relocations associated
## with it. The test checks that the update happens in the function that
## is not being optimized/relocated by BOLT.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --Ttext=0x80000
# RUN: llvm-bolt %t.exe --relocs -o %t.bolt --funcs=foo
# RUN: llvm-objdump -d --print-imm-hex %t.exe \
# RUN:   | FileCheck %s
# RUN: llvm-objdump -d --print-imm-hex %t.bolt \
# RUN:   | FileCheck %s --check-prefix=CHECK-POST-BOLT

  .text
  .globl foo
  .type foo,@function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo


  .text
  .globl _start
  .type _start,@function
_start:
  .cfi_startproc

## All four MOV instructions below are identical in the input binary, but
## different from each other after BOLT.
##
## foo value is 0x80000 pre-BOLT. Using relocations, llvm-bolt should correctly
## symbolize 0x80000 instruction operand and differentiate between an immediate
## constant 0x80000 and a reference to foo. When foo is relocated, BOLT should
## update references even from the code that is not being updated.

# CHECK:         80000 <foo>:

# CHECK:               <_start>:
# CHECK-POST-BOLT:     <_start>:

  movq $0x80000, 0x80000
# CHECK-NEXT:           movq $0x80000, 0x80000
# CHECK-POST-BOLT-NEXT: movq $0x80000, 0x80000

  movq $foo, 0x80000
# CHECK-NEXT:           movq $0x80000, 0x80000
# CHECK-POST-BOLT-NEXT: movq $0x[[#%x,ADDR:]], 0x80000

  movq $0x80000, foo
# CHECK-NEXT:           movq $0x80000, 0x80000
# CHECK-POST-BOLT-NEXT: movq $0x80000, 0x[[#ADDR]]

  movq $foo, foo
# CHECK-NEXT:           movq $0x80000, 0x80000
# CHECK-POST-BOLT-NEXT: movq $0x[[#ADDR]], 0x[[#ADDR]]

## After BOLT, foo is relocated after _start.

# CHECK-POST-BOLT: [[#ADDR]] <foo>:

  retq
  .size _start, .-_start
  .cfi_endproc
