# REQUIRES: loongarch
##
## pcalau12i+addi.d -> pcaddi relaxation (relaxPCHi20Lo12, isInt<22>) must not
## oscillate between remove=0 and remove=4. Without the fix, ld.lld reports
## "address assignment did not converge".
##
## Unlike RISC-V calls (8 -> 4 -> 2 bytes), every LoongArch pair/call36 site has
## only two states (8 or 4 bytes), so oscillation is always a 0 <-> 4 flip.
## The flip needs a site whose distance sits exactly at the +-2MiB limit while
## a ".p2align 4" (R_LARCH_ALIGN) absorbs some shrinks but not others, so the
## target address jitters between passes.
##
## The layout is deliberately fragile: the .space size below was found by
## searching a model of lld's relaxation loop. Do not "round" it.

# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax %s -o %t.o
# RUN: ld.lld -e _start %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## The two short-range sites in .text.a are always relaxed.
# CHECK-LABEL: <_start>:
# CHECK-NEXT:    pcaddi $t0,
# CHECK-NEXT:    pcaddi $t1,

.section .text.a,"ax"
.globl _start
_start:
  la.pcrel $t0, t_d
  la.pcrel $t1, t_e
  la.pcrel $t2, t_c            # forward, crosses the 2MiB filler
.globl t_d
t_d:
.globl t_e
t_e:

.section .text.b,"ax"
  .space 8
  la.pcrel $t3, t_a
  la.pcrel $t4, t_d            # backward, crosses the 2MiB filler
  .space 2097120    # 2MiB - 32: puts the sites right at the isInt<22> limit
  la.pcrel $t5, _start         # backward, near the limit
.globl t_c
t_c:
  .p2align 4        # R_LARCH_ALIGN; also makes .text.b 16-byte aligned
  la.pcrel $t6, t_d
.globl t_a
t_a:
