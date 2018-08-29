# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj -o %t.obj %s
# RUN: lld-link -lldmingw -entry:main %t.obj -out:%t.exe
# RUN: llvm-objdump -s %t.exe | FileCheck %s

.globl main
main:
  nop

.section .ctors.00005, "w"
  .quad 2
.section .ctors, "w"
  .quad 1
.section .ctors.00100, "w"
  .quad 3

.section .dtors, "w"
  .quad 4
.section .dtors.00100, "w"
  .quad 6
.section .dtors.00005, "w"
  .quad 5

# Also test that the .CRT section is merged into .rdata

.section .CRT$XCA, "dw"
  .quad 7
  .quad 8

# CHECK:      Contents of section .rdata:
# CHECK-NEXT: 140002000 07000000 00000000 08000000 00000000
# CHECK-NEXT: 140002010 01000000 00000000 02000000 00000000
# CHECK-NEXT: 140002020 03000000 00000000 04000000 00000000
# CHECK-NEXT: 140002030 05000000 00000000 06000000 00000000
