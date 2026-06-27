## Verify that BOLT correctly replaces branch targets during rewriting.
## This exercises replaceBranchTarget for conditional branches, unconditional
## branches, and compound compare-and-jump instructions.
##
## BOLT's fixBranches() calls replaceBranchTarget to update branch target
## symbols after CFG construction and optimization. The function must handle
## both extendable instructions (conditional/unconditional jumps, compound
## CJ) and gracefully skip non-extendable instructions (indirect jumps,
## dealloc_return). The latter case triggers an early return in
## replaceBranchTarget when getSymbolRefOperandNum returns false.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

##============================================================================
## Diamond CFG: conditional branch to .Ltrue, fallthrough to .Lfalse,
## both merge at .Ljoin. Tests that replaceBranchTarget correctly
## updates the conditional branch target after block layout.
##============================================================================
# CHECK-LABEL: <test_diamond>:
# CHECK:       p0 = cmp.eq(r0,#0x0)
# CHECK:       if (p0) jump:nt
# CHECK:       r0 = #0x2
# CHECK:       jump
# CHECK:       r0 = #0x1

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
  call test_diamond
  call test_multi_branch
  call test_compound_diamond
  jumpr r31
  .size _start, .-_start

  .globl test_diamond
  .type test_diamond,@function
  .p2align 4
test_diamond:
    p0 = cmp.eq(r0, #0)
    if (p0) jump:nt .Ltrue
.Lfalse:
    r0 = #2
    jump .Ljoin
.Ltrue:
    r0 = #1
.Ljoin:
    jumpr r31
  .size test_diamond, .-test_diamond

##============================================================================
## Function with conditional + unconditional branch. The conditional branch
## target and the unconditional jump target both go through
## replaceBranchTarget during fixBranches.
##============================================================================
# CHECK-LABEL: <test_multi_branch>:
# CHECK:       if (!p0) jump:nt
# CHECK:       r0 = #0xa
# CHECK:       jump
# CHECK:       r0 = #0x14

  .globl test_multi_branch
  .type test_multi_branch,@function
  .p2align 4
test_multi_branch:
    p0 = cmp.gt(r0, #5)
    if (!p0) jump:nt .Lsmall
.Lbig:
    r0 = #10
    jump .Lmb_done
.Lsmall:
    r0 = #20
.Lmb_done:
    jumpr r31
  .size test_multi_branch, .-test_multi_branch

##============================================================================
## Compound compare-and-jump (J4 CJ instruction) in a diamond. Compound
## CJ instructions are extendable (B9_PCREL) so replaceBranchTarget
## must update the symbol operand. The non-branch instructions in the
## same packet (r1 = r0) must not be affected.
##============================================================================
# CHECK-LABEL: <test_compound_diamond>:
# CHECK:       cmp.eq(r0,#0x0); if (p0.new) jump:nt
# CHECK:       r0 = #0x4
# CHECK:       jump
# CHECK:       r0 = #0x3

  .globl test_compound_diamond
  .type test_compound_diamond,@function
  .p2align 4
test_compound_diamond:
  {
    r1 = r0
    p0 = cmp.eq(r0, #0); if (p0.new) jump:nt .Lcd_true
  }
.Lcd_false:
    r0 = #4
    jump .Lcd_join
.Lcd_true:
    r0 = #3
.Lcd_join:
    jumpr r31
  .size test_compound_diamond, .-test_compound_diamond
