# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s

## The non-NC R_AARCH64_TLSLE_ADD_TPREL_LO12 relocation requires the full
## offset from the thread pointer to fit into an unsigned 12-bit immediate.
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -defsym=OOR=1 -o %t.bad.o
# RUN: not ld.lld %t.bad.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OOR --implicit-check-not=error:

# OOR: error: {{.*}}:(.text+0x4): relocation R_AARCH64_TLSLE_ADD_TPREL_LO12 out of range: 4096 is not in [0, 4095]; references 'v1'

.globl _start
_start:
 mrs x0, TPIDR_EL0
 add x0, x0, :tprel_lo12:v1

#CHECK: Disassembly of section .text:
#CHECK:      <_start>:
#CHECK-NEXT:   mrs     x0, TPIDR_EL0
#CHECK-NEXT:   add     x0, x0, #32

.section        .tbss,"awT",@nobits
.ifdef OOR
.space (0x1000 - 0x10)
.else
.space 0x10
.endif

.type   v1,@object
.globl  v1
v1:
.word  0
.size  v1, 4
