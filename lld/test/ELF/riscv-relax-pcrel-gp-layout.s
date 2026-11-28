# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax groups.s -o groups.32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax groups.s -o groups.64.o
# RUN: ld.lld --relax-gp --defsym '__global_pointer$=0x20800' --defsym foo=0x20000 --defsym bar=0x20fff -Ttext=0x10000 groups.32.o -o groups.32
# RUN: ld.lld --relax-gp --defsym '__global_pointer$=0x20800' --defsym foo=0x20000 --defsym bar=0x20fff -Ttext=0x10000 groups.64.o -o groups.64
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases groups.32 | FileCheck %s --check-prefix=GROUPS
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases groups.64 | FileCheck %s --check-prefix=GROUPS

# GROUPS-DAG: 00010000 g {{.*}}00000024 _start
# GROUPS-DAG: 00010004 l {{.*}} aligned
# GROUPS-DAG: 00010020 l {{.*}} target
# GROUPS:      <_start>:
# GROUPS-NEXT: lw a2, -0x800(gp)
# GROUPS:      <aligned>:
# GROUPS-NEXT: sw a2, 0x7ff(gp)
# GROUPS-NEXT: addi a3, gp, -0x800
# GROUPS-NEXT: jal ra, 0x10020 <target>
# GROUPS-NEXT: jal zero, 0x1001c <forward_high>
# GROUPS:      <forward_low>:
# GROUPS-NEXT: addi a4, gp, -0x800
# GROUPS-NEXT: jalr zero, 0x0(ra)
# GROUPS:      <forward_high>:
# GROUPS-NEXT: jal zero, 0x10014 <forward_low>
# GROUPS:      <target>:
# GROUPS-NEXT: jalr zero, 0x0(ra)

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax boundary.s -o boundary.32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax boundary.s -o boundary.64.o
# RUN: ld.lld --relax-gp --defsym '__global_pointer$=0x10000' --defsym zero=0 -Ttext=0x10000 boundary.32.o -o boundary.32
# RUN: ld.lld --relax-gp --defsym '__global_pointer$=0x10000' --defsym zero=0 -Ttext=0x10000 boundary.64.o -o boundary.64
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases boundary.32 | FileCheck %s --check-prefix=BOUNDARY
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases boundary.64 | FileCheck %s --check-prefix=BOUNDARY

# BOUNDARY-DAG: 00010000 g {{.*}}000007fc _start
# BOUNDARY-DAG: 000107f8 l {{.*}} target
# BOUNDARY:      <_start>:
# BOUNDARY-NEXT: addi a0, zero, 0x0
# BOUNDARY:      <.Lpcrel_hi1>:
# BOUNDARY-NEXT: addi a1, zero, 0x0
# BOUNDARY:      <.Lpcrel_hi2>:
# BOUNDARY-NEXT: addi a2, gp, 0x7f8
# BOUNDARY:      <target>:
# BOUNDARY-NEXT: jalr zero, 0x0(ra)

#--- groups.s
.globl _start
_start:
.Lhi0:
  auipc a0, %pcrel_hi(foo)
.Lhi1:
  auipc a1, %pcrel_hi(bar)
  lw a2, %pcrel_lo(.Lhi0)(a0)
  .p2align 2
aligned:
  sw a2, %pcrel_lo(.Lhi1)(a1)
  addi a3, a0, %pcrel_lo(.Lhi0)
  call target
  j forward_high
forward_low:
  addi a4, a0, %pcrel_lo(forward_high)
  ret
forward_high:
  auipc a0, %pcrel_hi(foo)
  j forward_low
target:
  ret
.size _start, .-_start

#--- boundary.s
.globl _start
_start:
  lla a0, zero
  lla a1, zero
  lla a2, target
  .space 0x804 - (. - _start)
target:
  ret
.size _start, .-_start
