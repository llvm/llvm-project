# REQUIRES: x86
## Test out-of-bounds section symbol offsets in SHF_MERGE sections.
## Non-section symbols and offset <= section_size are accepted, matching GNU ld.

# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64
# RUN: llvm-mc %s -o %t.err.o -filetype=obj -triple=x86_64 --defsym=ERR=1

## OOB section-symbol offsets diagnose (or warn).
# RUN: not ld.lld --threads=1 %t.err.o -o /dev/null -shared 2>&1 | FileCheck %s -DPREFIX=error --implicit-check-not=error:
# RUN: ld.lld --threads=1 %t.err.o -o /dev/null -shared --noinhibit-exec 2>&1 | FileCheck %s -DPREFIX=warning --implicit-check-not=warning:

## .foo is 8 bytes with entsize=8 (1 piece). .foo+8 (offset==size) is accepted.
# CHECK:      [[PREFIX]]: {{.*}}:(.foo): offset 0x9 is outside the section
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset 0x100000000 is outside the section
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset 0xffffffffffffffff is outside the section
## .rodata.str1.1 is "abc\0" (4 bytes). offset<=size is accepted.
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.rodata.str1.1): offset 0x5 is outside the section
## .data.retain references .foo-2 as well.
# CHECK-NEXT: [[PREFIX]]: {{.*}}:(.foo): offset 0xfffffffffffffffe is outside the section

## Test that --gc-sections with an out-of-bounds offset doesn't crash.
## .data is discarded but .data.retain (SHF_GNU_RETAIN) is kept.
## The bad offset prevents the piece from being marked live, so .foo is discarded.
# RUN: not ld.lld %t.err.o -o /dev/null --gc-sections 2>&1 | FileCheck %s --check-prefix=GC
# GC: error: relocation refers to a discarded section: .foo

# RUN: ld.lld -r %t.o -o %t.ro
# RUN: llvm-readelf -s -r %t.ro | FileCheck %s --check-prefix=RELOC

# RELOC:      R_X86_64_64 {{.*}} str_end + 0
# RELOC-NEXT: R_X86_64_64 {{.*}} fixed_end + 0

# RELOC:      0000000000000004 {{.*}} str_end
# RELOC:      0000000000000008 {{.*}} fixed_end

.globl _start
_start:

.data
.quad .foo + 8
.quad .rodata.str1.1 + 4
.quad str_end
.quad fixed_end

.ifdef ERR
.quad .foo + 9
.quad .foo + 0x100000000
.quad .foo - 1
.quad .rodata.str1.1 + 5

.quad a0 - 1
.quad a0 + 9

.section .data.retain,"awR"
.quad .foo - 2
.endif

.section	.foo,"aM",@progbits,8
a0:
.quad 0
.globl fixed_end
fixed_end:

.section	.rodata.str1.1,"aMS",@progbits,1
.asciz	"abc"
.globl str_end
str_end:
