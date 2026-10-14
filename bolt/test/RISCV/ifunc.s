## Check that BOLT recognizes a non-preemptible IFUNC IPLT entry after the
## linker canonicalizes the exported IFUNC symbol to that entry. Retain a local
## function symbol for the resolver, as in compiler-generated input. Moving the
## resolver also verifies that the IRELATIVE addend is updated.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -o %t.64.o %s
# RUN: ld.lld -q -o %t.64 %t.64.o
# RUN: llvm-readelf -Wr -Ws %t.64 | FileCheck %s --check-prefix=INPUT
# RUN: llvm-bolt %t.64 -o %t.64.bolt --use-old-text=0 --lite=0 \
# RUN:   --print-disasm --print-only=_start 2>&1 | FileCheck %s \
# RUN:   --check-prefix=BOLT \
# RUN:   --implicit-check-not="Expected BF to be presented as IFUNC resolver"
# RUN: llvm-readelf -Wr -Ws %t.64.bolt > %t.64.dump
# RUN: llvm-objdump -d --no-show-raw-insn %t.64.bolt >> %t.64.dump
# RUN: FileCheck %s --input-file=%t.64.dump \
# RUN:   --check-prefixes=ELF,IPLT,RESOLVER

# INPUT: R_RISCV_IRELATIVE {{ *}}[[#%x,INPUT_RESOLVER:]]
# INPUT: {{0*}}[[#INPUT_RESOLVER]] 4 FUNC LOCAL DEFAULT {{[0-9]+}} resolver
# INPUT: FUNC GLOBAL DEFAULT {{[0-9]+}} ifunc0

# BOLT: Binary Function "_start
# BOLT: auipc a0, %pcrel_hi("resolver/1@PLT")

# ELF: R_RISCV_IRELATIVE {{ *}}[[#%x,RESOLVER:]]
# ELF: {{[0-9a-f]+}} 4 FUNC {{.*}} func

# IPLT: Disassembly of section .iplt:
# IPLT: <ifunc0>:
# IPLT-NEXT: {{.*}} auipc t3,
# IPLT-NEXT: {{.*}} ld t3,

# RESOLVER: {{^ *}}[[#%x,RESOLVER]]:{{ *}}ret

  .text
  .globl _start
  .type _start, @function
_start:
1:
  auipc a0, %pcrel_hi(ifunc0)
  addi a0, a0, %pcrel_lo(1b)
  ret
  .size _start, .-_start

  .globl func
  .type func, @function
func:
  ret
  .size func, .-func

  .type resolver, @function
resolver:
  ret
  .size resolver, .-resolver

  .globl ifunc0
  .type ifunc0, @gnu_indirect_function
  .set ifunc0, resolver
