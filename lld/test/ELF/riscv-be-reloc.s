# REQUIRES: riscv
## Test that R_RISCV_32 data relocations produce big-endian output for
## big-endian RISC-V targets.

# RUN: rm -rf %t && mkdir %t && cd %t

## riscv32be: .word produces a BE 32-bit value
# RUN: llvm-mc -filetype=obj -triple=riscv32be -mattr=-relax %s -o 32.o
# RUN: ld.lld 32.o --defsym sym=0x11223344 -o out.32
# RUN: llvm-readobj -x .data out.32 | FileCheck --check-prefix=HEX32 %s

## Bytes should appear in big-endian order: 11 22 33 44
# HEX32: Hex dump of section '.data':
# HEX32: 11223344

## riscv64be: .word also produces a BE 32-bit value
# RUN: llvm-mc -filetype=obj -triple=riscv64be -mattr=-relax %s -o 64.o
# RUN: ld.lld 64.o --defsym sym=0x11223344 -o out.64
# RUN: llvm-readobj -x .data out.64 | FileCheck --check-prefix=HEX64 %s

## Bytes should appear in big-endian order: 11 22 33 44
# HEX64: Hex dump of section '.data':
# HEX64: 11223344

.global _start
_start:
    nop

.data
.word sym
