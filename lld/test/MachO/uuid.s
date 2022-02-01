# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem %t.o -o %t
# RUN: llvm-dwarfdump --uuid %t | FileCheck %s
# CHECK: 4C4C44{{([[:xdigit:]]{2})}}-5555-{{([[:xdigit:]]{4})}}-A1{{([[:xdigit:]]{2})}}-{{([[:xdigit:]]{12})}} (x86_64)

.globl _main
_main:
  ret
