# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax a.s -o rv64.o

# RUN: ld.lld --relax rv32.o lds -o rv32
# RUN: ld.lld --relax rv64.o lds -o rv64
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv32 | FileCheck %s
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv64 | FileCheck %s

# CHECK: 00101004 l       .secf1 {{0+}} f1
# CHECK: 00101004 l       .secf2 {{0+}} f2
# CHECK: 00101008 l       .secf2 {{0+}} f3
# CHECK: 00001000 g       .text  {{0+}} _start

# CHECK: <.callf1>
# CHECK-NEXT: auipc   ra, 256
# CHECK-NEXT: jalr    ra, 4(ra)

# CHECK: <.callf2>:
# CHECK-NEXT: auipc   ra, 256
# CHECK-NEXT: jalr    ra, -4(ra)

# CHECK: <.callf3>:
# CHECK-NEXT: auipc   ra, 256
# CHECK-NEXT: jalr    ra, -8(ra)

#--- a.s
.global _start
_start:
.section .callf1,"ax"
  call f1       # relax if there is a relaxation after

.section .callf2,"ax"
  call f2       # relax if there is no relaxation before
.section .callf3,"ax"
  call f3       # relax if there is no relaxation before
  .space 1048556

.section .secf1,"ax"
f1:
#  lui a0, 11

.section .secf2,"ax"
f2:
  lui a0, 12
f3:
  lui a0, 10

#--- lds
SECTIONS {
  .text 0x1000    : { }
  .callf1         : { }
  .callf2         : { }
  .callf3         : { }
  .secf1          : { }
  .secf2 0x101004 : { }
}
