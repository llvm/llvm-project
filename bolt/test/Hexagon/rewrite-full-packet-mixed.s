## Verify that BOLT can rewrite dense multi-instruction VLIW packets
## containing a mix of instruction types. Real Hexagon code commonly fills
## all 4 packet slots with ALU, load, store, and branch operations. This
## tests that the packet accumulation and flush logic in BinaryEmitter
## handles full packets correctly across different instruction combinations.
##
## Note: the assembler may reorder instructions within a packet for slot
## assignment, and may create duplex encodings for eligible pairs. The
## CHECK-DAG patterns allow any ordering within a packet.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
  call test_alu_load_branch
  call test_alu_store
  call test_triple_alu
  call test_load_alu_store
  jumpr r31
  .size _start, .-_start

##============================================================================
## Full 4-slot packet: 2 ALU + load + conditional branch.
## Tests that all 4 slots survive the BOLT round-trip in a packet
## containing ALU, memory, and control-flow operations.
## The assembler may reorder instructions to satisfy slot constraints.
##============================================================================
# CHECK-LABEL: <test_alu_load_branch>:
# CHECK-DAG:   add(r1,r2)
# CHECK-DAG:   and(r4,r5)
# CHECK-DAG:   memw(r7+#0x0)
# CHECK-DAG:   if (p0) jump:nt

  .globl test_alu_load_branch
  .type test_alu_load_branch,@function
  .p2align 4
test_alu_load_branch:
  {
    r0 = add(r1, r2)
    r3 = and(r4, r5)
    r6 = memw(r7 + #0)
    if (p0) jump:nt .Lhot_alb
  }
.Lcold_alb:
  r0 = #0
  jumpr r31
.Lhot_alb:
  r0 = #1
  jumpr r31
  .size test_alu_load_branch, .-test_alu_load_branch

##============================================================================
## 3-instruction packet: 2 ALU + store (no branch, falls through).
##============================================================================
# CHECK-LABEL: <test_alu_store>:
# CHECK-DAG:   add(r1,r2)
# CHECK-DAG:   sub(r4,r5)
# CHECK-DAG:   memw(r6+#0x0) = r7

  .globl test_alu_store
  .type test_alu_store,@function
  .p2align 4
test_alu_store:
  {
    r0 = add(r1, r2)
    r3 = sub(r4, r5)
    memw(r6 + #0) = r7
  }
  jumpr r31
  .size test_alu_store, .-test_alu_store

##============================================================================
## Full 4-slot packet: 3 ALU + unconditional jump.
##============================================================================
# CHECK-LABEL: <test_triple_alu>:
# CHECK-DAG:   add(r1,r2)
# CHECK-DAG:   and(r4,r5)
# CHECK-DAG:   or(r7,r8)
# CHECK:       jumpr r31

  .globl test_triple_alu
  .type test_triple_alu,@function
  .p2align 4
test_triple_alu:
  {
    r0 = add(r1, r2)
    r3 = and(r4, r5)
    r6 = or(r7, r8)
    jump .Ltriple_done
  }
.Ltriple_done:
  jumpr r31
  .size test_triple_alu, .-test_triple_alu

##============================================================================
## 3-instruction packet: load + ALU + store using different registers.
## A typical loop body pattern. The assembler may pack the load and
## store into a duplex instruction.
##============================================================================
# CHECK-LABEL: <test_load_alu_store>:
# CHECK-DAG:   add(r3,#0x1)
# CHECK-DAG:   memw(r2+#0x0)
# CHECK-DAG:   memw(r5+#0x0) = r6

  .globl test_load_alu_store
  .type test_load_alu_store,@function
  .p2align 4
test_load_alu_store:
  {
    r0 = memw(r2 + #0)
    r4 = add(r3, #1)
    memw(r5 + #0) = r6
  }
  jumpr r31
  .size test_load_alu_store, .-test_load_alu_store
