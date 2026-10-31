# REQUIRES: system-linux,bolt-runtime

## Counter spills and AMOs inside an LR/SC sequence can remove its eventual
## success guarantee. The forward diamond forces default spanning-tree
## instrumentation to place counters between LR and SC. Check a representative
## acquire/release loop and unmatched doubleword LR/SC instructions. Ordinary
## AMO instructions must remain instrumentable.
# RUN: %clang %cflags -Wl,-q -o %t.exe %s
# RUN: %t.exe
# RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata -o %t.instr
# RUN: llvm-objdump -d --no-show-raw-insn --disassemble-symbols=atomic_loop,lr_only,sc_only %t.instr \
# RUN:   | FileCheck %s --check-prefix=ATOMIC --implicit-check-not=amoadd
# RUN: llvm-objdump -d --no-show-raw-insn --disassemble-symbols=ordinary_amo %t.instr \
# RUN:   | FileCheck %s --check-prefix=AMO
# RUN: rm -f %t.fdata
# RUN: %t.instr
# RUN: FileCheck %s --check-prefix=PROFILE --input-file=%t.fdata
# RUN: llvm-bolt %t.exe --instrument --conservative-instrumentation --instrumentation-file=%t.cons.fdata -o %t.cons.instr
# RUN: llvm-objdump -d --no-show-raw-insn --disassemble-symbols=atomic_loop,lr_only,sc_only %t.cons.instr \
# RUN:   | FileCheck %s --check-prefix=ATOMIC --implicit-check-not=amoadd
# RUN: rm -f %t.cons.fdata
# RUN: %t.cons.instr
# RUN: FileCheck %s --check-prefix=PROFILE --input-file=%t.cons.fdata
# RUN: llvm-bolt %t.exe --data=%t.fdata --reorder-blocks=cache -o %t.bolt
# RUN: %t.bolt

## One AMO belongs to the input function and one updates its profile counter.
# AMO-LABEL: <ordinary_amo>:
# AMO-COUNT-2: amoadd.d
# AMO: ret
# PROFILE: ordinary_amo 0 0 1{{$}}

# ATOMIC-LABEL: <atomic_loop>:
# ATOMIC-NEXT: lr.w.aq a1, (a0)
# ATOMIC-NEXT: beqz a2,
# ATOMIC-NEXT: addi a1, a1, {{(0x)?1}}
# ATOMIC-NEXT: j
# ATOMIC-NEXT: addi a1, a1, {{(0x)?2}}
# ATOMIC-NEXT: sc.w.rl a3, a1, (a0)
# ATOMIC-NEXT: bnez a3,
# ATOMIC-NEXT: ret
# ATOMIC-LABEL: <lr_only>:
# ATOMIC-NEXT: lr.d.aqrl t0, (a0)
# ATOMIC-NEXT: ret
# ATOMIC-LABEL: <sc_only>:
# ATOMIC-NEXT: sc.d t0, t1, (a0)
# ATOMIC-NEXT: ret

  .option norvc
  .option norelax
  .text
  .globl main
  .type main,@function
main:
  addi sp, sp, -16
  sd ra, 0(sp)
  lla a0, value
  li a2, 1
  call atomic_loop
  li a2, 0
  call atomic_loop
  call ordinary_amo
  ld a0, 0(a0)
  addi a0, a0, -4
  snez a0, a0
  ld ra, 0(sp)
  addi sp, sp, 16
  ret
  .size main, .-main

  .globl atomic_loop
  .type atomic_loop,@function
atomic_loop:
.Lretry:
  lr.w.aq a1, (a0)
  beqz a2, .Lother
  addi a1, a1, 1
  j .Lstore
.Lother:
  addi a1, a1, 2
.Lstore:
  sc.w.rl a3, a1, (a0)
  bnez a3, .Lretry
  ret
  .size atomic_loop, .-atomic_loop

  .globl lr_only
  .type lr_only,@function
lr_only:
  lr.d.aqrl t0, (a0)
  ret
  .size lr_only, .-lr_only

  .globl sc_only
  .type sc_only,@function
sc_only:
  sc.d t0, t1, (a0)
  ret
  .size sc_only, .-sc_only

  .globl ordinary_amo
  .type ordinary_amo,@function
ordinary_amo:
  li a1, 1
  amoadd.d zero, a1, (a0)
  ret
  .size ordinary_amo, .-ordinary_amo

  .data
  .p2align 3
value:
  .quad 0

  .section .note.GNU-stack,"",@progbits
