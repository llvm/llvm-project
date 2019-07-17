# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: rm -f %/terr1.script
# RUN: echo  "\"" > %/terr1.script
# RUN: not ld.lld --vs-diagnostics  --version-script %/terr1.script -shared %/t.o -o %/t.so 2>&1 | \
# RUN: FileCheck %s -DSCRIPT="%/terr1.script"

# CHECK: [[SCRIPT]](1): error: [[SCRIPT]]:1: unclosed quote
