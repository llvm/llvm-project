# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump --no-show-raw-insn -d %t | FileCheck %s
# RUN: llvm-readobj --symbols %t | FileCheck --check-prefix=CHECK-SYM %s

## Test the local exec relocations that map to:
## R_AARCH64_TLSLE_MOVW_TPREL_G2
## R_AARCH64_TLSLE_MOVW_TPREL_G1
## R_AARCH64_TLSLE_MOVW_TPREL_G1_NC
## R_AARCH64_TLSLE_MOVW_TPREL_G0
## R_AARCH64_TLSLE_MOVW_TPREL_G0_NC
## They calculate the same value as the other TPREL relocations, namely the
## offset from the thread pointer TP. The G0, G1 and G2 refer to partitions
## of the result with G2 bits [47:32], G1 bits [31:16] and G0 bits [15:0]
## the NC variant does not check for overflow.
## In AArch64 the structure of the TLS at runtime is:
## | TCB | Alignment Padding | TLS Block |
## With TP pointing to the start of the TCB. All offsets will be positive.

.text
## Access variable in first partition
movz x0, #:tprel_g0:v0
## TCB + 0 == 16
# CHECK: mov     x0, #16

# CHECK-SYM:      Name: v0
# CHECK-SYM-NEXT: Value: 0x0

## Access variable in second partition
movz x0, #:tprel_g1:v1
movk x0, #:tprel_g0_nc:v1

## TCB + 65536 across movz and movk
# CHECK-NEXT: mov     x0, #65536
# CHECK-NEXT: movk    x0, #16

# CHECK-SYM:      Name: v1
# CHECK-SYM-NEXT: Value: 0x10000

## Access variable in third partition
movz x0, #:tprel_g2:v2
movk x0, #:tprel_g1_nc:v2
movk x0, #:tprel_g0_nc:v2

## TCB + 65536 + 4294967296 across movz and 2 movk instructions
# CHECK-NEXT: mov     x0, #4294967296
# CHECK-NEXT: movk    x0, #1, lsl #16
# CHECK-NEXT: movk    x0, #16

# CHECK-SYM:     Name: v2
# CHECK-SYM-NEXT:     Value: 0x100010000

.section .tbss,"awT",@nobits
.balign 16
v0:
.quad 0
.space 0x10000 - 8
v1:
.quad 0
.space 0x100000000 - 8
v2:
.quad 0
