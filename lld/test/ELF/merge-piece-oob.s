# REQUIRES: x86
## Test out-of-bounds section symbol offsets in SHF_MERGE sections.
## Non-section symbols and offset <= section_size are accepted, matching GNU ld.

# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64
# RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s -DPREFIX=error --implicit-check-not=error:
# RUN: ld.lld %t.o -o /dev/null -shared --noinhibit-exec 2>&1 | FileCheck %s -DPREFIX=warning --implicit-check-not=warning:

## .foo is 8 bytes with entsize=8 (1 piece). .foo+8 (offset==size) is accepted.
# CHECK:      [[PREFIX]]: {{.*}}:(.foo): offset 0x9 is outside the section
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset 0x100000000 is outside the section
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset 0xffffffffffffffff is outside the section
## .rodata.str1.1 is "abc\0" (4 bytes). offset<=size is accepted.
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.rodata.str1.1): offset 0x5 is outside the section
## .data.retain references .foo-1 as well.
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset 0xfffffffffffffffe is outside the section

## Test that --gc-sections with an out-of-bounds offset doesn't crash.
## .data is discarded but .data.retain (SHF_GNU_RETAIN) is kept.
## The bad offset prevents the piece from being marked live, so .foo is discarded.
# RUN: not ld.lld %t.o -o /dev/null --gc-sections 2>&1 | FileCheck %s --check-prefix=GC
# GC: error: relocation refers to a discarded section: .foo

.globl _start
_start:

.data
.quad .foo + 8
.quad .foo + 9
.quad .foo + 0x100000000
.quad .foo - 1
.quad .rodata.str1.1 + 3
.quad .rodata.str1.1 + 4
.quad .rodata.str1.1 + 5

.quad a0 - 1
.quad a0 + 9

.section .data.retain,"awR"
.quad .foo - 2

.section	.foo,"aM",@progbits,8
a0:
.quad 0

.section	.rodata.str1.1,"aMS",@progbits,1
.asciz	"abc"
