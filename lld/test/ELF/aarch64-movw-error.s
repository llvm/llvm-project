# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %t
# RUN: echo '.globl zero; zero = 0' | llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd -o %t2.o
# RUN: not ld.lld %t %t2.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: relocation R_AARCH64_MOVW_UABS_G0 out of range: 65536 is not in [0, 65535]
movn x0, #:abs_g0:zero+0x10000
# CHECK: relocation R_AARCH64_MOVW_UABS_G1 out of range: 4294967296 is not in [0, 4294967295]
movn x0, #:abs_g1:zero+0x100000000
# CHECK: relocation R_AARCH64_MOVW_UABS_G2 out of range: 281474976710656 is not in [0, 281474976710655]
movn x0, #:abs_g2:zero+0x1000000000000
# CHECK: relocation R_AARCH64_MOVW_SABS_G0 out of range: 65536 is not in [-65536, 65535]
movn x0, #:abs_g0_s:zero+0x10000
# CHECK: relocation R_AARCH64_MOVW_SABS_G1 out of range: 4294967296 is not in [-4294967296, 4294967295]
movn x0, #:abs_g1_s:zero+0x100000000
# CHECK: relocation R_AARCH64_MOVW_SABS_G2 out of range: 281474976710656 is not in [-281474976710656, 281474976710655]
movn x0, #:abs_g2_s:zero+0x1000000000000
# CHECK: relocation R_AARCH64_MOVW_SABS_G0 out of range: -65537 is not in [-65536, 65535]
movn x0, #:abs_g0_s:zero-0x10001
# CHECK: relocation R_AARCH64_MOVW_SABS_G1 out of range: -4295032832 is not in [-4294967296, 4294967295]
movn x0, #:abs_g1_s:zero-0x100010000
# CHECK: relocation R_AARCH64_MOVW_SABS_G2 out of range: -281479271677952 is not in [-281474976710656, 281474976710655]
movn x0, #:abs_g2_s:zero-0x1000100000000

# CHECK: relocation R_AARCH64_MOVW_PREL_G0 out of range: 65536 is not in [-65536, 65535]; references section '.text'
movn x0, #:prel_g0:.+0x10000
# CHECK: relocation R_AARCH64_MOVW_PREL_G1 out of range: 4294967296 is not in [-4294967296, 4294967295]
movn x0, #:prel_g1:.+0x100000000
# CHECK: relocation R_AARCH64_MOVW_PREL_G2 out of range: 281474976710656 is not in [-281474976710656, 281474976710655]
movn x0, #:prel_g2:.+0x1000000000000
# CHECK: relocation R_AARCH64_MOVW_PREL_G0 out of range: -65537 is not in [-65536, 65535]
movn x0, #:prel_g0:.-0x10001
# CHECK: relocation R_AARCH64_MOVW_PREL_G1 out of range: -4295032832 is not in [-4294967296, 4294967295]
movn x0, #:prel_g1:.-0x100010000
# CHECK: relocation R_AARCH64_MOVW_PREL_G2 out of range: -281479271677952 is not in [-281474976710656, 281474976710655]
movn x0, #:prel_g2:.-0x1000100000000

movz x0, #:tprel_g0: v1
# CHECK: relocation R_AARCH64_TLSLE_MOVW_TPREL_G0 out of range: 65552 is not in [-65536, 65535]; references 'v1'
movz x0, #:tprel_g1: v2
# CHECK: relocation R_AARCH64_TLSLE_MOVW_TPREL_G1 out of range: 4295032848 is not in [-4294967296, 4294967295]; references 'v2'
movz x0, #:tprel_g2: v3
# CHECK: relocation R_AARCH64_TLSLE_MOVW_TPREL_G2 out of range: 281479271743496 is not in [-281474976710656, 281474976710655]; references 'v3'

.section .tbss,"awT",@nobits
.balign 16
.space 0x10000
v1:
.quad 0
.space 0x100000000 - 8
v2:
.quad 0
.space 0x1000000000000 - 16
v3:
.quad 0
