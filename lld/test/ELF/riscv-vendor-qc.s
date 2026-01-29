# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+xqcili,+xqcibi,+xqcilb a.s -o rv32.o -riscv-add-build-attributes

# RUN: ld.lld rv32.o lds -pie -o rv32
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn %t/rv32 | FileCheck %s

#--- a.s

  .text
  .p2align 1

  .option exact

  .global _start
_start:

  # CHECK: qc.li t0, 0x0
  qc.li x5, %qc.abs20(test_zero)

  # CHECK: qc.li t0, -0x1
  qc.li x5, %qc.abs20(test_minus_one)

  # CHECK: qc.li t0, 0x7ffff
  qc.li x5, %qc.abs20(test_abs20_high)

  # CHECK: qc.li t0, -0x80000
  qc.li x5, %qc.abs20(test_abs20_low)

  # CHECK: qc.li t0, 0x12345
  qc.li x5, %qc.abs20(test_abs20_distinct)


  # CHECK: qc.e.li t1, 0x0
  qc.e.li x6, test_zero

  # CHECK: qc.e.li t1, -0x1
  qc.e.li x6, test_minus_one

  # CHECK: qc.e.li t1, 0x7fffffff
  qc.e.li x6, test_e32_high

  # CHECK: qc.e.li t1, -0x80000000
  qc.e.li x6, test_e32_low

  # CHECK: qc.e.li t1, 0x6789abcd
  qc.e.li x6, test_e32_distinct

  .global branch_case1
branch_case1:
  # CHECK: 11a0: qc.e.bgei t2, -0x8000, 0x11a0 <test_branch_here>
  qc.e.bgei x7, -0x8000, test_branch_here

  .global branch_case2
branch_case2:
  # CHECK: 11a6: qc.e.bltui t2, 0xffff, 0x12a6 <test_branch_high>
  qc.e.bltui x7, 0xffff, test_branch_high

  .global branch_case3
branch_case3:
  # CHECK: 11ac: qc.e.beqi t2, 0x7fff, 0x10ac <test_zero+0x10ac>
  qc.e.beqi x7, 0x7fff, test_branch_low

  .global branch_case4
branch_case4:
  # CHECK: 11b2: qc.e.bgeui t2, 0xffff, 0x126c <test_branch_distinct>
  qc.e.bgeui x7, 0xffff, test_branch_distinct


  .global call_case1
call_case1:
  # CHECK: 11b8: qc.e.j 0x11b8 <test_call_here>
  qc.e.j test_call_here

  .global call_case2
call_case2:
  # CHECK: 11be: qc.e.j 0x800011bc <test_e32_low+0x11bc>
  qc.e.j test_call_high

  .global call_case3
call_case3:
  # CHECK: 11c4: qc.e.j 0x800011c4 <test_minus_one+0xfffffffe800011c5>
  qc.e.j test_call_low

  .global call_case4
call_case4:
  # CHECK: 11ca: qc.e.j 0x23457954 <test_abs20_high+0x233d7955>
  qc.e.j test_call_distinct

  # CHECK: 11d0: qc.e.j 0x0 <test_zero>
  qc.e.j test_zero

  # CHECK: 11d6: qc.e.j 0xfffffffe <test_minus_one+0xfffffffeffffffff>
  qc.e.j test_call_abs_high

#--- lds

test_zero = 0;
test_minus_one = -1;

test_abs20_high = 0x7ffff;
test_abs20_low = -0x80000;
test_abs20_distinct = 0x12345;

test_e32_high = 0x7fffffff;
test_e32_low = -0x80000000;
test_e32_distinct = 0x6789abcd;

test_branch_here = branch_case1;
test_branch_high = branch_case2 + 0x100;
test_branch_low = branch_case3 - 0x100;
test_branch_distinct = branch_case4 + 0xba;

test_call_here = call_case1;
test_call_high = (call_case2 + 0x7ffffffe) & 0xffffffff;
test_call_low = (call_case3 - 0x80000000) | (0xffffffff << 32);
test_call_distinct = (call_case4 + 0x12345678a) & 0xffffffff;
test_call_abs_high = 0xfffffffffffffffe;
