## Check that a relocation against a data symbol with a negative addend is not
## mistaken for a reference into a function when "symbol + addend" happens to
## resolve inside one. Compilers fold a bias into the displacement for indexed
## accesses, e.g. "mov sym-0x3fe00(,%rax,8)", and the unbiased address is never
## used on its own.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --nostdlib -e _start --image-base=0x200000 \
# RUN:   --section-start=.text=0x200000 --section-start=.mydata=0x400000
# RUN: llvm-bolt %t.exe -o %t.bolt --relocs 2>&1 | FileCheck %s

## The biased displacement resolves into the middle of an instruction in
## "target". BOLT used to report that as an external branch and ignore both
## functions.
# CHECK-NOT: corrupted control flow

  .text
  .globl target
  .type target, @function
target:
## 10-byte instruction at 0x200000, so 0x200003 is not an instruction boundary.
  movabsq $0x1122334455667788, %rax
  retq
  .size target, .-target

  .globl _start
  .type _start, @function
_start:
## datasym is at 0x400000, so datasym-0x1ffffd resolves to 0x200003, inside
## "target" above.
  movq datasym-0x1ffffd(,%rax,8), %r14
  retq
  .size _start, .-_start

  .section .mydata, "aw", @progbits
  .globl datasym
datasym:
  .quad 0
