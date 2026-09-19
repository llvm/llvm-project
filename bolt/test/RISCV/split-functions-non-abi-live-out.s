## Without --assume-abi, registers that are not ABI return registers may still
## be live at a function exit. Check that BOLT does not use such a register for
## a cross-fragment long jump.

# RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.o %s
# RUN: ld.lld --emit-relocs -e _start -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt -split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1 2>&1 | FileCheck %s
# RUN: llvm-readelf -S %t.bolt | FileCheck --check-prefix=SECTIONS %s
# RUN: llvm-objdump -d --section=.text %t.bolt | \
# RUN:   FileCheck --check-prefix=DISASM %s

# CHECK: BOLT-WARNING: keeping _start unsplit: no dead register for a RISC-V long jump
# SECTIONS-NOT: .text.cold
# DISASM-LABEL: <_start>:
# DISASM: li t0, 0x7
# DISASM-NOT: auipc

  .text
  .globl _start
  .type _start, @function
_start:
  li t0, 7
  beq a0, zero, .Lcold
  ret
.Lcold:
  ret
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE
