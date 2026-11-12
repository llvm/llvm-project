# REQUIRES: arm

# RUN: llvm-mc -triple thumbv6m-arm-eabi --filetype=obj %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d %t --no-show-raw-insn | FileCheck %s --check-prefixes=CHECK,CHECK-LE

# RUN: llvm-mc -triple thumbebv6m-arm-eabi --filetype=obj %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d %t --no-show-raw-insn | FileCheck %s --check-prefixes=CHECK,CHECK-BE

# CHECK-LE:                    file format elf32-littlearm
# CHECK-BE:                    file format elf32-bigarm

# CHECK:                       Disassembly of section .text:

# CHECK-LABEL: [[#%x,TARGET:]] <target>:
# CHECK-NEXT:      [[#TARGET]]: bx lr

# CHECK-LABEL:                 <_start>:
# CHECK-NEXT:                   b      0x[[#TARGET]] <target>
# CHECK-NEXT:                   beq    0x[[#TARGET]] <target>

    .thumb
    .section .text.1, "ax", %progbits
target:
    bx lr

    .section .text.2, "ax", %progbits
    .globl _start
_start:
    b.n target   // R_ARM_THM_JUMP11
    beq.n target // R_ARM_THM_JUMP8
