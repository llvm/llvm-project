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

## --no-relax disables all linker relaxation, including table jump.
## Verify the .riscv.jvt section is not created and no cm.jt/cm.jalt emitted.
# RUN: ld.lld %t.rv32.o --no-relax --relax-tbljal --defsym foo=0x150000 --defsym foo_1=0x150010 --defsym foo_3=0x150030 -o %t.norelax
# RUN: llvm-readelf -S %t.norelax | FileCheck --check-prefix=NORELAX-SEC %s
# RUN: llvm-objdump -d --mattr=+zcmt --no-show-raw-insn %t.norelax | FileCheck --check-prefix=NORELAX %s
# NORELAX-SEC-NOT: .riscv.jvt
# NORELAX-NOT:     cm.jt
# NORELAX-NOT:     cm.jalt

## RV32 -- 21 calls to foo become cm.jalt at index 0x20 (first cm.jalt slot).
## Tails become cm.jt with indices assigned by saved-bytes descending, tiebreaking
## on symbol name: foo_3 (saved=30) -> 0x0, foo_1 (saved=18) -> 0x1,
## foo and foo_2 both saved=6 -> 0x2 and 0x3 (alphabetical).
# RV32-NOT:         cm.jt
# RV32-NOT:         cm.jalt
# RV32-COUNT-21:    cm.jalt 0x20
# RV32-NEXT:        cm.jt   0x2
# RV32-NEXT:        cm.jt   0x1
# RV32-NEXT:        cm.jt   0x1
# RV32-NEXT:        cm.jt   0x1
# RV32-NEXT:        cm.jt   0x0
# RV32-NEXT:        c.j
# RV32-NEXT:        cm.jt   0x0
# RV32-NEXT:        cm.jt   0x0
# RV32-NEXT:        cm.jt   0x0
# RV32-NEXT:        cm.jt   0x0
# RV32-NOT:         cm.jt
# RV32-NOT:         cm.jalt

## RV64 -- no cm.jalt (CMJALT padding cost with 8-byte entries is too high).
## Only tails are relaxed to cm.jt: foo_3 -> 0x0, foo_1 -> 0x1.
# RV64-NOT:         cm.jalt
# RV64:             cm.jt   0x1
# RV64-NEXT:        cm.jt   0x1
# RV64-NEXT:        cm.jt   0x1
# RV64-NEXT:        cm.jt   0x0
# RV64-NEXT:        c.j
# RV64-NEXT:        cm.jt   0x0
# RV64-NEXT:        cm.jt   0x0
# RV64-NEXT:        cm.jt   0x0
# RV64-NEXT:        cm.jt   0x0
# RV64-NOT:         cm.jt
# RV64-NOT:         cm.jalt

## Verify table entries contain the target addresses (little-endian).
## RV32 CMJT -- foo_3 (idx 0), foo_1 (idx 1), foo (idx 2), foo_2 (idx 3);
## then padding to idx 32; then CMJALT for foo.
# JVT32: 30001500 10001500 00001500
# JVT32: 00001500

## RV64 CMJT -- foo_3 (idx 0), foo_1 (idx 1). No CMJALT.
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
