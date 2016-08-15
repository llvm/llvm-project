# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { \
# RUN:  . = 0x1000; .aaa : ONLY_IF_RO { *(.aaa.*) } \
# RUN:  . = 0x2000; .aaa : ONLY_IF_RW { *(.aaa.*) } } " > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | FileCheck %s

# CHECK:      Sections:
# CHECK-NEXT: Idx Name          Size      Address          Type
# CHECK-NEXT:   0               00000000 0000000000000000
# CHECK-NEXT:   1 .aaa          00000010 0000000000002000 DATA
# CHECK-NEXT:   2 .text         00000001 0000000000002010 TEXT DATA
# CHECK-NEXT:   3 .symtab       00000030 0000000000000000
# CHECK-NEXT:   4 .shstrtab     00000026 0000000000000000
# CHECK-NEXT:   5 .strtab       00000008 0000000000000000

# RUN: echo "SECTIONS { \
# RUN:  . = 0x1000; .aaa : ONLY_IF_RW { *(.aaa.*) } \
# RUN:  . = 0x2000; .aaa : ONLY_IF_RO { *(.aaa.*) } } " > %t2.script
# RUN: ld.lld -o %t2 --script %t2.script %t
# RUN: llvm-objdump -section-headers %t2 | FileCheck %s --check-prefix=REV

# REV:      Sections:
# REV-NEXT: Idx Name          Size      Address          Type
# REV-NEXT:   0               00000000 0000000000000000
# REV-NEXT:   1 .aaa          00000010 0000000000001000 DATA
# REV-NEXT:   2 .text         00000001 0000000000002000 TEXT DATA
# REV-NEXT:   3 .symtab       00000030 0000000000000000
# REV-NEXT:   4 .shstrtab     00000026 0000000000000000
# REV-NEXT:   5 .strtab       00000008 0000000000000000

.global _start
_start:
 nop

.section .aaa.1, "aw"
.quad 1

.section .aaa.2, "aw"
.quad 1
