# REQUIRES: riscv
## Test R_RISCV_CALL_PLT relocations with big-endian RISC-V targets.

# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32be -mattr=-relax %s -o 32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64be -mattr=-relax %s -o 64.o

# RUN: ld.lld 32.o --defsym foo=_start+8 --defsym bar=_start -o out.32
# RUN: ld.lld 64.o --defsym foo=_start+8 --defsym bar=_start -o out.64
# RUN: llvm-objdump -d --no-show-raw-insn out.32 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn out.64 | FileCheck %s
# CHECK:      auipc   ra, 0x0
# CHECK-NEXT: jalr    0x8(ra)
# CHECK:      auipc   ra, 0x0
# CHECK-NEXT: jalr    -0x8(ra)

.global _start
_start:
    call    foo
    call    bar
