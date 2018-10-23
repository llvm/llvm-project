# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: ld.lld -e A %t -o /dev/null \
# RUN:   -noinhibit-exec -icf=all 2>&1 | FileCheck %s

    .section    .text.C,"ax",@progbits
    .globl  C
C:
    mov poppy, %rax
    retq

B = 0x1234

    .section    .text.A,"ax",@progbits
    .globl  A
A:
    mov poppy, %rax
    retq

    .cg_profile A, B, 100
    .cg_profile A, C, 40
    .cg_profile B, C, 30
    .cg_profile adena1, A, 30
    .cg_profile A, adena2, 30
    .cg_profile poppy, A, 30

# CHECK: unable to order absolute symbol: B

# RUN: ld.lld %t -o /dev/null \
# RUN:   -noinhibit-exec -icf=all --no-warn-symbol-ordering 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NOWARN
# NOWARN-NOT: unable to order
