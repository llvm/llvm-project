# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o

## Make sure we got the right relocations.
# RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=REL
# REL: R_HEX_B9_PCREL         00000000   b9
# REL: R_HEX_B13_PCREL        00000000   b13
# REL: R_HEX_B15_PCREL        00000000   b15
# REL: R_HEX_B22_PCREL        00000000   b22

# RUN: ld.lld %t.o -o %t.out --section-start=.text=0x1000000 \
# RUN:  --section-start=b9=0x1000400 --section-start=b13=0x1004000 \
# RUN:  --section-start=b15=0x1010000 --section-start=b22=0x1800000 \
# RUN:  --threads=1
# RUN: llvm-objdump -d --no-show-raw-insn %t.out | FileCheck %s

# CHECK-NOT: trampoline
# CHECK: 01000000 <_start>:
# CHECK-NEXT: 1000000: {  nop }
# CHECK-NEXT: 1000004: {  r0 = #0x0 ; jump 0x1000400 }
# CHECK-NEXT: 1000008: {  if (r0==#0) jump:t 0x1004000 }
# CHECK-NEXT: 100000c: {  if (p0) jump:nt 0x1010000 }
# CHECK-NEXT: 1000010: {  jump 0x1800000 }

 .globl _start
 .type _start, @function
_start:
## Make sure the first jump is within range
nop
{ r0 = #0; jump #b9 }
if (r0==#0) jump:t #b13
if (p0) jump #b15
jump #b22

.section b9, "ax"
nop

.section b13, "ax"
nop

.section b15, "ax"
nop

.section b22, "ax"
nop
