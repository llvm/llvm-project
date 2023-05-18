# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:   .text 0x10000: { *(.text) } \
# RUN:   .data 0x200010000 : { *(.data) } \
# RUN: }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: relocation R_PPC64_PCREL34 out of range: 8589934592 is not in [-8589934592, 8589934591]; references section '.data'
	plwa 3, glob_overflow@PCREL(0), 1

# CHECK-NOT: relocation
	plwa 3, .data@PCREL(0), 1

.data
glob_overflow:
	.long	0
	.size	glob_overflow, 4
