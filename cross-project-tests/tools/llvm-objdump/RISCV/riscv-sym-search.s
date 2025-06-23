# RUN: %clang --target=fuchsia-elf-riscv64 -march=rv64g %s -nostdlib -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK:   auipc a0, 0x101
# CHECK:   ld a0, 0x8(a0) <ldata+0x1000>
.global _start
.text
_start:
  la a0, gdata

.skip 0x100000
ldata:
  .int 0

.data
gdata:
  .int 0
