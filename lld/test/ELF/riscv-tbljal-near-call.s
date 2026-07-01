# REQUIRES: riscv

## Test that the saved-bytes estimation for table jump candidates accounts for
## call relaxation and R_RISCV_JAL: when a R_RISCV_CALL/CALL_PLT target is within
## ±1MiB, the scanning part should consider about this extra byte-savings.
## For R_RISCV_JAL, it should always be 2 bytes.
##
## Also covers the c.jal case in scanTableJumpEntries: on RV32, a near
## call (rd=ra) within ±2KiB will relax to c.jal, so it should not be added
## as a cm.jalt candidate.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+zcmt %s -o %t.o
# RUN: ld.lld %t.o --relax-tbljal --defsym near=0x90000 --defsym near_jal=0x90010 --defsym far=0x200000 -o %t
# RUN: llvm-objdump -d -M no-aliases --mattr=+zcmt --no-show-raw-insn %t | FileCheck %s

## "far" tails should be relaxed to cm.jt.
## While "near" should NOT be cm.jt because its actual saving is only 2 bytes
## (the tail first relaxes to jal), which is less than wordsize (4).
## It should remain as jal (relaxed from auipc+jalr by relaxCall).
## That's the same for R_RISCV_JAL, whose saving should only be 2 bytes.
##
## "near_call" is within c.jal range on RV32 (±2KiB), so the
## call should relax to c.jal instead of cm.jalt.

# CHECK-COUNT-20: cm.jt
# CHECK-NEXT: jal zero
# CHECK-NEXT: jal zero
# CHECK-NEXT: c.jal
# CHECK-NOT:  cm.jalt

.global _start
.p2align 3
_start:
  .rept 20
  tail far
  .endr
  tail near
  jal zero, near_jal
  call near_call

## near_call is within c.jal range (±2KiB) from the call site.
.p2align 2
near_call:
  ret
