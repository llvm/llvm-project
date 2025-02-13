# RUN: llvm-mc -triple thumbv6m-arm-eabi --filetype=obj %s -o %t.obj
# RUN: ld.lld %t.obj -o %t.linked
# RUN: llvm-objdump -d %t.linked | FileCheck %s --check-prefixes=CHECK,CHECK-LE

# RUN: llvm-mc -triple thumbebv6m-arm-eabi --filetype=obj %s -o %t.obj
# RUN: ld.lld %t.obj -o %t.linked
# RUN: llvm-objdump -d %t.linked | FileCheck %s --check-prefixes=CHECK,CHECK-BE

# CHECK-LE:                    file format elf32-littlearm
# CHECK-BE:                    file format elf32-bigarm

# CHECK:                       Disassembly of section .text:

# CHECK-LABEL: [[#%x,TARGET:]] <target>:
# CHECK-NEXT:      [[#TARGET]]: 4770    bx lr

# CHECK-LABEL:                 <_start>:
# CHECK-NEXT:                   e7fd    b      0x[[#TARGET]] <target>
# CHECK-NEXT:                   d0fc    beq    0x[[#TARGET]] <target>

    .thumb
    .section .text.1, "ax", %progbits
target:
    bx lr
 
    .section .text.2, "ax", %progbits
    .globl _start
_start:
    b.n target   // R_ARM_THM_JUMP11
    beq.n target // R_ARM_THM_JUMP8
