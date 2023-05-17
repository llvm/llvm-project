# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo "SECTIONS { symbol2 = symbol; }" > %t2.script
# RUN: not ld.lld -o /dev/null -T %t2.script %t.o 2>&1 \
# RUN:  | FileCheck -check-prefix=ERR %s
# ERR: {{.*}}.script:1: symbol not found: symbol

.global _start
_start:
 nop
