# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
# RUN: echo "MEMORY {                                       \
# RUN:   REGION (rwx) : ORIGIN = 0x1000, LENGTH = 0x100     \
# RUN: }                                                    \
# RUN:                                                      \
# RUN: SECTIONS {                                           \
# RUN:   .aaa ORIGIN(REGION) + 0x8 : { *(.aaa) } > REGION   \
# RUN:   _stext = .;                                        \
# RUN:   .bbb ORIGIN(REGION) : { *(.bbb) } > REGION         \
# RUN:   . = _stext;                                        \
# RUN: }" > %t.script
# RUN: ld.lld %t --script %t.script -o %t2
# RUN: llvm-objdump -section-headers %t2 | FileCheck %s
# CHECK: .aaa        00000008 0000000000001008 DATA
# CHECK: .bbb        00000008 0000000000001000 DATA

.section .aaa, "a"
.quad 0

.section .bbb, "a"
.quad 0
