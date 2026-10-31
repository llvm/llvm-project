## Entry tail patches can relax while staying within their eight-byte budget.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -Ttext=0x10fc0 %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --force-patch --no-huge-pages --use-gnu-stack --custom-allocation-vma=0x11000 --align-text=4 --align-functions=4 -o %t.bolt
# RUN: llvm-objdump -d --no-show-raw-insn --section=.bolt.org.text -M no-aliases %t.bolt | FileCheck %s --check-prefix=PATCH
# RUN: llvm-mc -triple=riscv64 -mattr=+c -defsym=COMPRESSED=1 -filetype=obj %s -o %t.c.o
# RUN: ld.lld --emit-relocs -Ttext=0x10fc0 %t.c.o -o %t.c.exe
# RUN: llvm-bolt %t.c.exe --force-patch --no-huge-pages --use-gnu-stack --custom-allocation-vma=0x11000 --align-text=4 --align-functions=4 -o %t.c.bolt
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.c.bolt | FileCheck %s --check-prefix=COMPRESSED

# PATCH-LABEL: <_start.org.0>:
# PATCH-NEXT: jal zero, 0x11000 <_start>
# PATCH:      10fc8: ecall
# COMPRESSED-LABEL: <_start.org.0>:
# COMPRESSED-NEXT: c.j 0x11000 <_start>

## The two-byte trailing function cannot hold an entry patch. Its reference
## must follow the emitted _start when scanning it after _start was patched.
# COMPRESSED-LABEL: <tiny>:
# COMPRESSED-NEXT: c.j 0x11000 <_start>

  .option norvc
  .text
  .globl _start
  .type _start,@function
_start:
  li a0, 0
  li a7, 93
  ecall
  .reloc 0, R_RISCV_NONE
  .size _start, .-_start

  .ifdef COMPRESSED
  .option rvc
  .option exact
  .globl tiny
  .type tiny,@function
tiny:
  c.j _start
  .size tiny, .-tiny
  .endif
