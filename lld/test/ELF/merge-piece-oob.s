# REQUIRES: x86
## Test out-of-bounds section symbol offsets in SHF_MERGE sections.

# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64
# RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s -DPREFIX=error --implicit-check-not=error:
# RUN: ld.lld %t.o -o /dev/null -shared --noinhibit-exec 2>&1 | FileCheck %s -DPREFIX=warning --implicit-check-not=warning:

# CHECK:      [[PREFIX]]: {{.*}}:(.foo): offset is outside the section
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset is outside the section
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset is outside the section
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset is outside the section
## .rodata.str1.1 is "abc\0" (4 bytes).
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.rodata.str1.1): offset is outside the section
## .data.retain references .foo-1 as well.
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset is outside the section

.globl _start
_start:

.data
.quad .foo + 8
.quad .foo + 9
.quad .foo + 0x100000000
.quad .foo - 1
.quad .rodata.str1.1 + 3
.quad .rodata.str1.1 + 4

.quad a0 - 1
.quad a0 + 9

.section .data.retain,"awR"
.quad .foo - 1

.section	.foo,"aM",@progbits,8
a0:
.quad 0

.section	.rodata.str1.1,"aMS",@progbits,1
.asciz	"abc"
