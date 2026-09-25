## IFUNC aliases can share a resolver while using distinct IPLT/GOT entries.
## Preserve the linked target of both CALL_PLT and CALL, including tail calls.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -o %t.o %s
# RUN: ld.lld --emit-relocs -o %t %t.o
# RUN: llvm-readelf -Wr %t | FileCheck %s --check-prefix=RELOCS
# RUN: llvm-bolt %t -o %t.bolt --use-old-text=0 --lite=0
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt | FileCheck %s --check-prefix=CALLS

# RELOCS: R_RISCV_IRELATIVE {{ *}}[[#%x,RESOLVER:]]
# RELOCS: R_RISCV_IRELATIVE {{ *}}[[#RESOLVER]]
# RELOCS: R_RISCV_CALL_PLT {{0*}}[[#RESOLVER]] ifunc0 + 0
# RELOCS: R_RISCV_CALL {{0*}}[[#RESOLVER]] ifunc1 + 0

# CALLS-LABEL: <_start>:
# CALLS:       jalr {{.*}} <.iplt>
# CALLS:       jalr {{.*}} <.iplt+0x10>
# CALLS:       jr {{.*}} <.iplt+0x10>

  .text
  .option norelax
  .globl _start
  .type _start, @function
_start:
  call ifunc0
  .reloc ., R_RISCV_CALL, ifunc1
  auipc ra, 0
  jalr ra
  tail ifunc1
  .size _start, .-_start

  .type resolver, @function
resolver:
  ret
  .size resolver, .-resolver

  .globl ifunc0, ifunc1
  .type ifunc0, @gnu_indirect_function
  .set ifunc0, resolver
  .type ifunc1, @gnu_indirect_function
  .set ifunc1, resolver
