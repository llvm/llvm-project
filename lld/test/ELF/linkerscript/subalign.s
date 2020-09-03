# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o

# RUN: echo "SECTIONS { .aaa : { *(.aaa.*) } }" > %t1.script
# RUN: ld.lld -o %t1 --script %t1.script %t1.o
# RUN: llvm-objdump -s %t1 | FileCheck --check-prefix=NOALIGN %s
# NOALIGN:      Contents of section .aaa:
# NOALIGN-NEXT:   01000000 00000000 00000000 00000000
# NOALIGN-NEXT:   00000000 00000000 00000000 00000000
# NOALIGN-NEXT:   02000000 00000000 00000000 00000000
# NOALIGN-NEXT:   00000000 00000000 00000000 00000000
# NOALIGN-NEXT:   03000000 00000000 00000000 00000000
# NOALIGN-NEXT:   00000000 00000000 00000000 00000000
# NOALIGN-NEXT:   00000000 00000000 00000000 00000000
# NOALIGN-NEXT:   00000000 00000000 00000000 00000000
# NOALIGN-NEXT:   04000000 00000000

# RUN: echo "SECTIONS { .aaa : SUBALIGN(1) { *(.aaa.*) } }" > %t2.script
# RUN: ld.lld -o %t2 --script %t2.script %t1.o
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=SUBALIGN %s
# SUBALIGN: Contents of section .aaa:
# SUBALIGN:   01000000 00000000 02000000 00000000
# SUBALIGN:   03000000 00000000 04000000 00000000

## Test we do not assert or crash when dot(.) is used inside SUBALIGN.
## Value of dot is undefined. Some versions of ld.bfd do not allow to use dot
## in such expressions.
# RUN: echo "SECTIONS { . = 0x32; .aaa : SUBALIGN(.) { *(.aaa*) } }" > %t3.script
# RUN: not ld.lld %t1.o --script %t3.script -o /dev/null 2>&1 | FileCheck --check-prefix=ERR1 %s
# ERR1: {{.*}}.script:1: unable to get location counter value

## Test we are able to link with zero alignment, this is consistent with bfd 2.26.1.
# RUN: echo "SECTIONS { .aaa : SUBALIGN(0) { *(.aaa*) } }" > %t4.script
# RUN: ld.lld %t1.o --script %t4.script -o %t4
# RUN: llvm-objdump -s %t4 | FileCheck --check-prefix=SUBALIGN %s

## Test we fail gracefully when alignment value is not a power of 2.
# RUN: echo "SECTIONS { .aaa : SUBALIGN(3) { *(.aaa*) } }" > %t5.script
# RUN: not ld.lld %t1.o --script %t5.script -o /dev/null 2>&1 | FileCheck --check-prefix=ERR2 %s
# ERR2: {{.*}}.script:1: alignment must be power of 2

.global _start
_start:
 nop

.section .aaa.1, "a"
.align 16
.quad 1

.section .aaa.2, "a"
.align 32
.quad 2

.section .aaa.3, "a"
.align 64
.quad 3

.section .aaa.4, "a"
.align 128
.quad 4
