# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t --icf=all --ignore-data-address-equality --print-icf-sections | FileCheck %s --check-prefix ICF
# RUN: llvm-objdump -s -d -print-imm-hex %t | FileCheck %s

# ICF: selected section <internal>:(.rodata)
# ICF-NEXT: removing identical section <internal>:(.rodata)

# CHECK: {{^}}.text:
# CHECK-NEXT: movq 0x[[ADDR:[0-9a-f]+]], %rax
# CHECK-NEXT: movq 0x[[ADDR]], %rax
# CHECK: Contents of section .rodata:
# CHECK-NEXT: 2a000000 00000000 67452301 10325476

.section .rodata, "a"
  .quad 42

.section .rodata.cst4,"aM",@progbits,4
rodata4:
  .long 0x01234567
  .long 0x76543210
  .long 0x01234567
  .long 0x76543210

.section .rodata.cst8,"aM",@progbits,8
rodata8:
  .long 0x01234567
  .long 0x76543210

.section .text,"ax"
  movq rodata4, %rax
  movq rodata8, %rax
