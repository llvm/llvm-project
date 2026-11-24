# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s

## Like R_AARCH64_TLSLE_ADD_TPREL_LO12, the non-NC LDST variants require the
## full offset from the thread pointer to fit into an unsigned 12-bit
## immediate; the imm12 field of the scaled load/store instruction holds bits
## 11:log2(access-size) of the offset.
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -defsym=OOR=1 -o %t.bad.o
# RUN: not ld.lld %t.bad.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OOR --implicit-check-not=error:

# OOR: error: {{.*}}:(.text+0x4): relocation R_AARCH64_TLSLE_LDST8_TPREL_LO12 out of range: 4096 is not in [0, 4095]; references 'v1'
# OOR: error: {{.*}}:(.text+0x8): relocation R_AARCH64_TLSLE_LDST16_TPREL_LO12 out of range: 4096 is not in [0, 4095]; references 'v1'
# OOR: error: {{.*}}:(.text+0xc): relocation R_AARCH64_TLSLE_LDST32_TPREL_LO12 out of range: 4096 is not in [0, 4095]; references 'v1'
# OOR: error: {{.*}}:(.text+0x10): relocation R_AARCH64_TLSLE_LDST64_TPREL_LO12 out of range: 4096 is not in [0, 4095]; references 'v1'
# OOR: error: {{.*}}:(.text+0x14): relocation R_AARCH64_TLSLE_LDST128_TPREL_LO12 out of range: 4096 is not in [0, 4095]; references 'v1'

## An in-range but misaligned offset must be rejected by the alignment check
## the non-NC cases share with their _NC siblings.
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -defsym=MISALIGNED=1 -o %t.mis.o
# RUN: not ld.lld %t.mis.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISALIGN --implicit-check-not=error:

# MISALIGN: error: {{.*}}:(.text+0x18): improper alignment for relocation R_AARCH64_TLSLE_LDST16_TPREL_LO12: 0x11 is not aligned to 2 bytes

.globl _start
_start:
 mrs x0, TPIDR_EL0
 ldrb w0, [x0, :tprel_lo12:v1]
 ldrh w1, [x1, :tprel_lo12:v1]
 ldr  w2, [x2, :tprel_lo12:v1]
 ldr  x3, [x3, :tprel_lo12:v1]
 ldr  q4, [x4, :tprel_lo12:v1]
.ifdef MISALIGNED
 ldrh w7, [x7, :tprel_lo12:v2]
.endif

#CHECK: Disassembly of section .text:
#CHECK:      <_start>:
#CHECK-NEXT:   mrs     x0, TPIDR_EL0
#CHECK-NEXT:   ldrb    w0, [x0, #4080]
#CHECK-NEXT:   ldrh    w1, [x1, #4080]
#CHECK-NEXT:   ldr     w2, [x2, #4080]
#CHECK-NEXT:   ldr     x3, [x3, #4080]
#CHECK-NEXT:   ldr     q4, [x4, #4080]

.section        .tbss,"awT",@nobits
.p2align 4
## v2 sits at an odd offset (0x11) from the thread pointer for the
## misaligned-offset check above.
.byte 0
.type   v2,@object
.globl  v2
v2:
.byte   0
.size   v2, 1
.p2align 4

## v1 sits at 0xff0 (4080) from the thread pointer: bit 11 of the offset is
## set, so a wrong shift or mask in the scaled immediate shows up in the
## disassembly above. In the OOR variant it sits at 0x1000 (4096), one past
## the 12-bit limit.
.ifdef OOR
.space (0xff0 - 0x10)
.else
.space (0xfe0 - 0x10)
.endif

.type   v1,@object
.globl  v1
v1:
.space  16
.size   v1, 16
