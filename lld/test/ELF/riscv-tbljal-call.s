# REQUIRES: riscv

## Test that call/tail instructions are relaxed to cm.jt/cm.jalt when
## --relax-tbljal is enabled and the table jump is profitable.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+zcmt %s -o %t.rv64.o
# RUN: ld.lld %t.rv32.o --relax-tbljal --defsym foo=0x150000 --defsym foo_1=0x150010 --defsym foo_3=0x150030 -o %t.rv32
# RUN: ld.lld %t.rv64.o --relax-tbljal --defsym foo=0x150000 --defsym foo_1=0x150010 --defsym foo_3=0x150030 -o %t.rv64

## Check disassembly for cm.jalt (rd=ra) and cm.jt (rd=zero).
# RUN: llvm-objdump -d -M no-aliases --mattr=+zcmt --no-show-raw-insn %t.rv32 | FileCheck --check-prefix=RV32 %s
# RUN: llvm-objdump -d -M no-aliases --mattr=+zcmt --no-show-raw-insn %t.rv64 | FileCheck --check-prefix=RV64 %s

## Check jump table contents.
# RUN: llvm-readelf -x .riscv.jvt %t.rv32 | FileCheck --check-prefix=JVT32 %s
# RUN: llvm-readelf -x .riscv.jvt %t.rv64 | FileCheck --check-prefix=JVT64 %s

## 21 calls to foo become cm.jalt (RV32), tails become cm.jt.
# RV32-COUNT-21: cm.jalt
# RV32:         cm.jt
# RV32:         cm.jt
# RV32:         cm.jt
# RV32:         cm.jt

# RV64:         cm.jt
# RV64:         cm.jt
# RV64:         cm.jt

## Verify table entries contain the target addresses (little-endian).
# JVT32: 30001500 10001500 00001500
# JVT64: 30001500 00000000 10001500 00000000

.global _start
.p2align 3
_start:
  .rept 21
  call foo
  .endr
  tail foo
  tail foo_1
  tail foo_1
  tail foo_1
  tail foo_3
  tail foo_2
  tail foo_3
  tail foo_3
  tail foo_3
  tail foo_3

foo_2:
  nop
