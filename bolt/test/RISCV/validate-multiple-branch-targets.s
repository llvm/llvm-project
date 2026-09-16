## Reject every invalid target from a source, including an instruction-middle
## target after a data-island target. Deferring setIgnored() until validation
## completes must preserve the instruction boundaries needed for later checks.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -Ttext=0x10000 %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s --check-prefix=DIAG
# RUN: llvm-nm --defined-only %t.bolt | FileCheck %s --check-prefix=ADDR --implicit-check-not=__ENTRY_
# RUN: llvm-bolt %t.exe -o %t.skip.bolt --skip-funcs=_start 2>&1 | FileCheck %s --check-prefix=DIAG
# RUN: llvm-nm --defined-only %t.skip.bolt | FileCheck %s --check-prefix=ADDR --implicit-check-not=__ENTRY_

# DIAG: BOLT-WARNING: corrupted control flow detected in function _start: an external branch/call targets an invalid instruction in function data_target at address 0x10004; ignoring both functions
# DIAG-NEXT: BOLT-WARNING: ignoring entry point at address 0x10004 in constant island of function data_target
# DIAG-NEXT: BOLT-WARNING: corrupted control flow detected in function _start: an external branch/call targets an invalid instruction in function instruction_target at address 0x1000e; ignoring both functions
# DIAG-NOT: corrupted control flow detected
# DIAG-NOT: ignoring entry point
# ADDR-DAG: 0000000000010000 T data_target
# ADDR-DAG: 000000000001000c T instruction_target
# ADDR-DAG: 0000000000010014 T _start

## Targets precede the source so their instruction boundaries are available
## when the explicit --skip-funcs run scans the source's external references.
  .globl data_target
  .type data_target, @function
data_target:
  nop
.Ldata:
  .word 0xcd027057
  ret
  .size data_target, .-data_target

  .globl instruction_target
  .type instruction_target, @function
instruction_target:
  addi a0, a0, 1
  ret
  .size instruction_target, .-instruction_target

  .globl _start
  .type _start, @function
_start:
  beq a0, a1, .Ldata
  beq a2, a3, instruction_target + 2
  ret
  .size _start, .-_start
