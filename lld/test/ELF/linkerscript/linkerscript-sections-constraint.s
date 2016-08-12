# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { \
# RUN:  .writable : ONLY_IF_RW { *(.writable) } \
# RUN:  .readable : ONLY_IF_RO { *(.readable) }}" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | \
# RUN:   FileCheck -check-prefix=BASE %s
# BASE: Sections:
# BASE-NEXT: Idx Name          Size      Address          Type
# BASE-NEXT:   0               00000000 0000000000000000
# BASE-NEXT:   1 .writable     00000004 0000000000000200 DATA
# BASE-NEXT:   2 .readable     00000004 0000000000000204 DATA

# RUN: echo "SECTIONS { \
# RUN:  .foo : ONLY_IF_RO { *(.foo.*) } \
# RUN:  .writable : ONLY_IF_RW { *(.writable) } \
# RUN:  .readable : ONLY_IF_RO { *(.readable) }}" > %t2.script
# RUN: ld.lld -o %t2 --script %t2.script %t
# RUN: llvm-objdump -section-headers %t2 | \
# RUN:   FileCheck -check-prefix=NO1 %s
# NO1: Sections:
# NO1-NEXT: Idx Name          Size      Address          Type
# NO1-NEXT: 0               00000000 0000000000000000
# NO1-NEXT: 1 .writable     00000004 0000000000000200 DATA
# NO1-NEXT: 2 .readable     00000004 0000000000000204 DATA
# NO1-NEXT: 3 .text         00000001 0000000000000208 TEXT DATA
# NO1-NEXT: 4 .foo.2        00000004 0000000000000209 DATA
# NO1-NEXT: 5 .foo.1        00000004 000000000000020d TEXT DATA

.global _start
_start:
  nop

.section .writable, "aw"
writable:
 .long 1

.section .readable, "a"
readable:
 .long 2

.section .foo.1, "awx"
 .long 0

.section .foo.2, "aw"
 .long 0
