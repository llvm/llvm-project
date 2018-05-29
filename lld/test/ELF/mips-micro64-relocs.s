# REQUIRES: mips

# Check handling of some microMIPS relocations in 64-bit mode.

# RUN: llvm-mc -filetype=obj -triple=mips64el-unknown-linux \
# RUN:         -mattr=micromips %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips64el-unknown-linux \
# RUN:         -mattr=micromips %S/Inputs/mips-dynamic.s -o %t2.o
# RUN: ld.lld %t1.o %t2.o -o %t.exe
# RUN: llvm-objdump -d %t.exe | FileCheck %s

  .global  __start
__start:
  lui     $7,  %highest(_foo+0x300047FFF7FF8)
  lui     $7,  %higher (_foo+0x300047FFF7FF8)
  lui     $gp, %hi(%neg(%gp_rel(__start)))
  lui     $gp, %lo(%neg(%gp_rel(__start)))

# CHECK:      20000:  a7 41 03 00  lui $7, 3
# CHECK-NEXT: 20004:  a7 41 05 00  lui $7, 5
# CHECK-NEXT: 20008:  bc 41 02 00  lui $gp, 2
# CHECK-NEXT: 2000c:  bc 41 00 80  lui $gp, 32768
