# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
## Use --threads=1 to keep emitted warnings across sections sequential.
# RUN: not ld.lld %t.o -o /dev/null --threads=1 2>&1 | FileCheck --implicit-check-not "out of range" %s

	.globl	_start
	.type	_start, @function
_start:

# CHECK: relocation R_HEX_B9_PCREL out of range: 1028 is not in [-1024, 1023]
{r0 = #0; jump #1f}
.space (1<<10)
.section b9, "ax"
1:

# CHECK: relocation R_HEX_B13_PCREL out of range: 16388 is not in [-16384, 16383]
if (r0==#0) jump:t #1f
.space (1<<14)
.section b13, "ax"
1:

# CHECK: relocation R_HEX_B15_PCREL out of range: 65540 is not in [-65536, 65535]
if (p0) jump #1f
.space (1<<16)
.section b15, "ax"
1:

# CHECK: relocation R_HEX_B22_PCREL out of range: 8388612 is not in [-8388608, 8388607]
jump #1f
.space (1<<23)
.section b22, "ax"
1:
