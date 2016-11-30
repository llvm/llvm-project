# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { }" > %t.script
# RUN: not ld.lld %t.o -script %t.script -o %t 2>&1 | FileCheck %s
# CHECK: error: {{.*}}:(.text+0x0): undefined symbol '_edata'
# CHECK: error: {{.*}}:(.text+0x8): undefined symbol '_etext'
# CHECK: error: {{.*}}:(.text+0x10): undefined symbol '_end'

.global _start,_end,_etext,_edata
.text
_start:
 .quad _edata + 0x1
 .quad _etext + 0x1
 .quad _end + 0x1

.data
  .word 1
.bss
  .align 4
  .space 6
