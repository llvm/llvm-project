## Verify that BOLT can build a CFG for functions with J4 compound
## compare-and-jump instructions. These instructions have the branch
## target in a different operand position than J2 jumps; BOLT uses
## TSFlags-based getExtendableOp to find the target generically.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe --print-cfg --print-only=test_compound -o /dev/null \
# RUN:   > %t.log 2>&1
# RUN: FileCheck %s --input-file=%t.log

# CHECK: Binary Function "test_compound" after building cfg
# CHECK: if (p0.new) jump:t .Ltmp

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call test_compound
    jumpr r31
  .size _start, .-_start

  .globl test_compound
  .type test_compound,@function
  .p2align 4
test_compound:
  {
    p0 = cmp.eq(r0, #0)
    if (p0.new) jump:t .Ltarget
  }
    r0 = #1
    jumpr r31
.Ltarget:
    r0 = #0
    jumpr r31
  .size test_compound, .-test_compound
