# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { /DISCARD/ : { *(.aaa*) } }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t 2>&1 | FileCheck --check-prefix=WARN %s
# RUN: llvm-objdump -section-headers %t1 | FileCheck %s

# WARN: relocation refers to discarded section {{.+}}(.aaa)
# CHECK-NOT: .aaa

.section .aaa,"a"
aab:
  .quad 0

.section .zzz,"a"
  .quad aab
