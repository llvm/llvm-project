# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=CHECK,RELAX
# RUN: ld.lld --no-relax %t.o -o %t.norelax
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.norelax | FileCheck %s --check-prefixes=CHECK,NORELAX
# RUN: llvm-readobj -S -r %t | FileCheck -check-prefix=RELOC %s

#Local-Dynamic to Local-Exec relax creates no
#RELOC:      Relocations [
#RELOC-NEXT: ]

## Reject local-exec TLS relocations for -shared.
# RUN: not ld.lld -shared %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ERR: error: relocation R_AARCH64_TLSLE_ADD_TPREL_HI12 against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_ADD_TPREL_LO12_NC against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_ADD_TPREL_LO12 against v3 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_ADD_TPREL_HI12 against v2 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_ADD_TPREL_LO12_NC against v2 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_ADD_TPREL_HI12 against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_ADD_TPREL_HI12 against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_ADD_TPREL_HI12 against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_LDST8_TPREL_LO12 against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_LDST16_TPREL_LO12 against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_LDST32_TPREL_LO12 against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_LDST64_TPREL_LO12 against v1 cannot be used with -shared
# ERR: error: relocation R_AARCH64_TLSLE_LDST128_TPREL_LO12 against v1 cannot be used with -shared

.globl _start
_start:
 mrs x0, TPIDR_EL0
 add x0, x0, :tprel_hi12:v1
 add x0, x0, :tprel_lo12_nc:v1
 mrs x0, TPIDR_EL0
 add x0, x0, :tprel_lo12:v3
 mrs x0, TPIDR_EL0
 add x0, x0, :tprel_hi12:v2
 add x0, x0, :tprel_lo12_nc:v2
 add x2, x1, :tprel_hi12:v1
 add w3, w3, :tprel_hi12:v1
 add sp, sp, :tprel_hi12:v1
 ldrb w0, [x0, :tprel_lo12:v1]
 ldrh w1, [x1, :tprel_lo12:v1]
 ldr w2, [x2, :tprel_lo12:v1]
 ldr x3, [x3, :tprel_lo12:v1]
 ldr q4, [x4, :tprel_lo12:v1]

# TCB size = 0x10 and v1 is first element from TLS register.
#CHECK: Disassembly of section .text:
#CHECK:      <_start>:
#CHECK-NEXT:   mrs     x0, TPIDR_EL0
#RELAX-NEXT:   nop
#NORELAX-NEXT:   add     x0, x0, #0, lsl #12
#CHECK-NEXT:   add     x0, x0, #16
#CHECK-NEXT:   mrs     x0, TPIDR_EL0
#CHECK-NEXT:   add     x0, x0, #20
#CHECK-NEXT:   mrs     x0, TPIDR_EL0
#CHECK-NEXT:   add     x0, x0, #4095, lsl #12
#CHECK-NEXT:   add     x0, x0, #4088
#CHECK-NEXT:   add     x2, x1, #0, lsl #12
#CHECK-NEXT:   add     w3, w3, #0, lsl #12
#RELAX-NEXT:   nop
#NORELAX-NEXT:   add     sp, sp, #0, lsl #12
#CHECK-NEXT:   ldrb    w0, [x0, #16]
#CHECK-NEXT:   ldrh    w1, [x1, #16]
#CHECK-NEXT:   ldr     w2, [x2, #16]
#CHECK-NEXT:   ldr     x3, [x3, #16]
#CHECK-NEXT:   ldr     q4, [x4, #16]

.section        .tbss,"awT",@nobits

.type   v1,@object
.globl  v1
.p2align 2
v1:
.word  0
.size  v1, 4

.type   v3,@object
.globl  v3
.p2align 2
v3:
.word  0
.size  v3, 4

## The current offset from the thread pointer is 24. Raise it to just below the
## 24-bit limit.
.space (0xfffff8 - 24)

.type   v2,@object
.globl  v2
.p2align 2
v2:
.word  0
.size  v2, 4
