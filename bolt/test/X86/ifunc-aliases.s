## Sharing an IFUNC resolver does not imply sharing an IPLT/GOT entry.

# RUN: llvm-mc -filetype=obj -triple=x86_64 -o %t.o %s
# RUN: ld.lld --emit-relocs -o %t %t.o
# RUN: llvm-bolt %t -o %t.bolt --use-old-text=0 --lite=0
# RUN: llvm-objdump -d --section=.iplt --no-show-raw-insn %t.bolt > %t.dump
# RUN: llvm-objdump -d --disassemble-symbols=_start --no-show-raw-insn %t.bolt >> %t.dump
# RUN: FileCheck %s --input-file=%t.dump

# CHECK:       [[#%x,IPLT:]] <.iplt>:
# CHECK-LABEL: <_start>:
# CHECK:       callq 0x[[#IPLT]]
# CHECK:       callq 0x[[#IPLT+16]]
# CHECK:       jmp 0x[[#IPLT+16]]

  .text
  .globl _start
  .type _start, @function
_start:
  call ifunc0
  call ifunc1
  jmp ifunc1
  .size _start, .-_start

  .type resolver, @function
resolver:
  retq
  .size resolver, .-resolver

  .globl ifunc0, ifunc1
  .type ifunc0, @gnu_indirect_function
  .set ifunc0, resolver
  .type ifunc1, @gnu_indirect_function
  .set ifunc1, resolver
