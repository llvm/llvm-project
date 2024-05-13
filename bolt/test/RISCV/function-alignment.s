## Test that BOLT uses a minimum function alignment of 4 (or 2 for RVC) bytes.

# RUN: llvm-mc -triple=riscv64 -filetype=obj -o %t.o %s
# RUN: ld.lld -q -o %t %t.o
# RUN: llvm-bolt --align-functions=1 --use-old-text=0 -o %t.bolt %t
# RUN: llvm-nm -n %t.bolt | FileCheck %s

# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj -o %t-c.o %s
# RUN: ld.lld -q -o %t-c %t-c.o
# RUN: llvm-bolt --align-functions=1 --use-old-text=0 -o %t-c.bolt %t-c
# RUN: llvm-nm -n %t-c.bolt | FileCheck --check-prefix=CHECK-C %s

# CHECK:      {{[048c]}} T _start
# CHECK-NEXT: {{[048c]}} T dummy

# CHECK-C:      {{[02468ace]}} T _start
# CHECK-C-NEXT: {{[02468ace]}} T dummy

    .text

    # Make sure input binary is only 1 byte aligned. BOLT should increase the
    # alignment to 2 or 4 bytes.
    .byte 0
    .balign 1

    .globl _start
    .type _start, @function
_start:
    # Dummy reloc to force relocation mode.
    .reloc 0, R_RISCV_NONE
    ret
    .size _start, .-_start

    .globl dummy
    .type dummy, @function
dummy:
    ret
    .size dummy, .-dummy
