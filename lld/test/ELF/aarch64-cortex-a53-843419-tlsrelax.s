// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t.o
// RUN: ld.lld -fix-cortex-a53-843419 %t.o -o %t2
// RUN: llvm-objdump --triple=aarch64-linux-gnu -d %t2 | FileCheck %s

// The following code sequence is covered by the TLS IE to LE relaxation. It
// transforms the ADRP, LDR to MOVZ, MOVK. The former can trigger a
// cortex-a53-843419 patch, whereas the latter can not. As both
// relaxation and patching transform instructions very late in the
// link there is a possibility of them both being simultaneously
// applied. In this case the relaxed sequence is immune from the erratum so we
// prefer to keep it.
 .text
 .balign 4096
 .space  4096 - 8
 .globl _start
 .type  _start,@function
_start:
 mrs    x1, tpidr_el0
 adrp   x0, :gottprel:v
 ldr    x1, [x0, #:gottprel_lo12:v]
 adrp   x0, :gottprel:v
 ldr    x1, [x0, #:gottprel_lo12:v]
 ret

// CHECK: <_start>:
// CHECK-NEXT:   211ff8:        d53bd041        mrs     x1, TPIDR_EL0
// CHECK-NEXT:   211ffc:        d2a00000        movz    x0, #0, lsl #16
// CHECK-NEXT:   212000:        f2800201        movk    x1, #16
// CHECK-NEXT:   212004:        d2a00000        movz    x0, #0, lsl #16
// CHECK-NEXT:   212008:        f2800201        movk    x1, #16
// CHECK-NEXT:   21200c:        d65f03c0        ret

 .type  v,@object
 .section       .tbss,"awT",@nobits
 .globl v
v:
 .word 0
